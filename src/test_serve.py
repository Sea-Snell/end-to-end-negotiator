import requests
import json
import multiprocessing as mp

def call_model(x):
    headers, payload = x
    session = requests.Session()
    return json.loads(session.post('http://10.150.0.5:5000/api/message',headers=headers,data=payload).text)

def submit_model(x):
    headers, payload = x
    session = requests.Session()
    return json.loads(session.post('http://10.150.0.5:5000/api/submit',headers=headers,data=payload).text)


if __name__ == "__main__":
    headers = {'User-Agent': 'Mozilla/5.0'}
    utterances = []
    ctx = ['1', '10', '3', '10', '1', '10']
    choice = None

    while choice is None:
        payload = {'history': json.dumps(utterances), 'ctx': json.dumps(ctx)}
        with mp.Pool(processes=1) as pool:
            utterances, choice = pool.map(call_model, [(headers, payload)])[0]
        print(utterances, choice)
    print(submit_model((headers, payload)))