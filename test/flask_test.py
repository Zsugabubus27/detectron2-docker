import time
import requests

st = time.time()
resp = requests.get('http://localhost:5555')

print(resp.text, resp.url, time.time() - st)

def print_url(r, *args, **kwargs):
	print('printURL func', r.url)

st = time.time()
resp = requests.post('http://localhost:5555/detect', hooks={'response': print_url})
print(time.time() - st)
#print(resp.text, resp.url, time.time() - st)