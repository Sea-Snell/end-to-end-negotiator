# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from flask import Flask, request
from flask_cors import CORS
import argparse
import pdb
import re
import random

import numpy as np
import torch
from torch import optim
from torch import autograd
import torch.nn as nn

from agent import *
import utils
from utils import ContextGenerator
from dialog import Dialog, DialogLogger
from models.rnn_model import RnnModel
from models.latent_clustering_model import LatentClusteringPredictionModel, BaselineClusteringModel
import domain
import redis
import json
import time
import pickle as pkl
import traceback
import multiprocessing as mp

Q = None

app = Flask(__name__)
CORS(app)

r = redis.Redis(host='localhost', port=6379, db=0)

# class SelfPlay(object):
#     def __init__(self, dialog, ctx_gen, args, logger=None):
#         self.dialog = dialog
#         self.ctx_gen = ctx_gen
#         self.args = args
#         self.logger = logger if logger else DialogLogger()

#     def run(self):
#         n = 0
#         for ctxs in self.ctx_gen.iter():
#             n += 1
#             if self.args.smart_alice and n > 1000:
#                 break
#             self.logger.dump('=' * 80)
#             self.dialog.run(ctxs, self.logger)
#             self.logger.dump('=' * 80)
#             self.logger.dump('')
#             if n % 100 == 0:
#                 self.logger.dump('%d: %s' % (n, self.dialog.show_metrics()), forced=True)

def flip(raw_dialogue):
    new_dialogue = []
    for speaker, utterance in raw_dialogue:
        if speaker == 'YOU':
            new_dialogue.append(('THEM', utterance))
        elif speaker == 'THEM':
            new_dialogue.append(('YOU', utterance))
        else:
            raise NotImplementedError
    return new_dialogue

def fetch_response(agent, raw_ctx, raw_dialogue):
    agent.feed_context(raw_ctx)
    for speaker, utterance in raw_dialogue:
        assert speaker == 'YOU' or speaker == 'THEM'
        you = (speaker == 'YOU')
        agent.read(list(map(lambda x: x.strip(), utterance.split())) + ['<eos>'], you=you)
    response = ' '.join(agent.write())
    if '<selection>' in response:
        return raw_dialogue + [('YOU', response.strip())], agent.choose()
    response = response[:response.find('<eos>')].strip()
    return raw_dialogue + [('YOU', response)], None

def fetch_submission(agent, raw_ctx, raw_dialogue):
    agent.feed_context(raw_ctx)
    for speaker, utterance in raw_dialogue:
        assert speaker == 'YOU' or speaker == 'THEM'
        you = (speaker == 'YOU')
        # might need to not include <eos> on submission event
        agent.read_silent(list(map(lambda x: x.strip(), utterance.split())) + ['<eos>'], you=you)
    return agent.choose()

def get_agent_type(model, smart=False):
    if isinstance(model, LatentClusteringPredictionModel):
        if smart:
            return LatentClusteringRolloutAgent
        else:
            return LatentClusteringAgent
    elif isinstance(model, RnnModel):
        if smart:
            return RnnRolloutAgent
        else:
            return RnnAgent
    elif isinstance(model, BaselineClusteringModel):
        if smart:
            return BaselineClusteringRolloutAgent
        else:
            return BaselineClusteringAgent
    else:
        assert False, 'unknown model type: %s' % (model)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/message', methods=['GET', 'POST'])
def respond():
    if request.method == 'GET':
        history = request.args.get('history', None)
        ctx = request.args.get('ctx', None)
    else:
        history = request.form.get('history', None)
        ctx = request.form.get('ctx', None)
    history = json.loads(history)
    ctx = json.loads(ctx)
    print('[DEBUG] History recieved:', history)
    # generate response
    request_id = int(r.incr('request_id_counter'))
    print('[DEBUG] queueing message with request id:', request_id)
    Q.put((request_id, ctx, history, 'response'))
    while not r.exists("result_%d" % (request_id)):
        time.sleep(0.05)
    print('[DEBUG] de-queueing message with request id:', request_id)
    result = pkl.loads(r.get("result_%d" % (request_id)))
    r.delete("result_%d" % (request_id))
    print('[DEBUG] Response:', result)
    return json.dumps(result)

@app.route('/api/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'GET':
        history = request.args.get('history', None)
        ctx = request.args.get('ctx', None)
    else:
        history = request.form.get('history', None)
        ctx = request.form.get('ctx', None)
    history = json.loads(history)
    ctx = json.loads(ctx)
    print('[DEBUG] History recieved:', history)
    # generate response
    request_id = int(r.incr('request_id_counter'))
    print('[DEBUG] queueing message with request id:', request_id)
    Q.put((request_id, ctx, history, 'submit'))
    while not r.exists("result_%d" % (request_id)):
        time.sleep(0.05)
    print('[DEBUG] de-queueing message with request id:', request_id)
    result = pkl.loads(r.get("result_%d" % (request_id)))
    r.delete("result_%d" % (request_id))
    print('[DEBUG] Response:', result)
    return json.dumps(result)

def _chatbot_f(agent, ctx, history, kind):
    if kind == 'submit':
        return fetch_submission(agent, ctx, history)
    elif kind == 'response':
        return fetch_response(agent, ctx, history)
    else:
        raise NotImplementedError

def model_process(agent):
    print('CHATBOT LOADED!')
    while True:
        try:
            request_id, ctx, history, kind = Q.get()
            result = _chatbot_f(agent, ctx, history, kind)
            r.set('result_%d' % (request_id), pkl.dumps(result))
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

def flask_process(port):
    app.run(host='0.0.0.0', port=port, threaded=True, processes=1)

def main():
    global Q
    parser = argparse.ArgumentParser(description='selfplaying script')
    parser.add_argument('--alice_model_file', type=str,
        help='Alice model file')
    parser.add_argument('--alice_forward_model_file', type=str,
        help='Alice forward model file')
    parser.add_argument('--bob_model_file', type=str,
        help='Bob model file')
    parser.add_argument('--context_file', type=str,
        help='context file')
    parser.add_argument('--temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--pred_temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--verbose', action='store_true', default=False,
        help='print out converations')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--score_threshold', type=int, default=6,
        help='successful dialog should have more than score_threshold in score')
    parser.add_argument('--max_turns', type=int, default=20,
        help='maximum number of turns in a dialog')
    parser.add_argument('--log_file', type=str, default='',
        help='log successful dialogs to file for training')
    parser.add_argument('--smart_alice', action='store_true', default=False,
        help='make Alice smart again')
    parser.add_argument('--diverse_alice', action='store_true', default=False,
        help='make Alice smart again')
    parser.add_argument('--rollout_bsz', type=int, default=3,
        help='rollout batch size')
    parser.add_argument('--rollout_count_threshold', type=int, default=3,
        help='rollout count threshold')
    parser.add_argument('--smart_bob', action='store_true', default=False,
        help='make Bob smart again')
    parser.add_argument('--selection_model_file', type=str,  default='',
        help='path to save the final model')
    parser.add_argument('--rollout_model_file', type=str,  default='',
        help='path to save the final model')
    parser.add_argument('--diverse_bob', action='store_true', default=False,
        help='make Alice smart again')
    parser.add_argument('--ref_text', type=str,
        help='file with the reference text')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--domain', type=str, default='object_division',
        help='domain for the dialogue')
    parser.add_argument('--visual', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--eps', type=float, default=0.0,
        help='eps greedy')
    parser.add_argument('--data', type=str, default='data/negotiate',
        help='location of the data corpus')
    parser.add_argument('--unk_threshold', type=int, default=20,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--validate', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--port', type=int, default=5000)

    args = parser.parse_args()

    utils.use_cuda(args.cuda)
    utils.set_seed(args.seed)

    model = utils.load_model(args.alice_model_file)
    ty = get_agent_type(model, args.smart_alice)
    alice = ty(model, args, name='Alice', train=False, diverse=args.diverse_alice)

    # test_ctx = ['1', '10', '3', '10', '1', '10']
    # test_dialogue = []
    # while len(test_dialogue) == 0 or '<selection>' not in test_dialogue[-1][1]:
    #     test_dialogue, choice = fetch_response(alice, test_ctx, test_dialogue)
    #     print(test_dialogue, choice)
    #     test_dialogue = flip(test_dialogue)
    # print(fetch_submission(alice, test_ctx, test_dialogue))

    q = mp.Manager().Queue()
    Q = q

    p = mp.Process(target=flask_process, args=(args.port,))
    p.start()

    model_process(alice)




if __name__ == '__main__':
    main()
