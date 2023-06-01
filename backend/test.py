import json
import requests

url = 'http://apis.juhe.cn/tyfy/query'

param = {"word": "希望", "type": "同义词", "key": "a9f946f996eadf163b94c33f53512687"}
header = {"Content-Type": "application/x-www-form-urlencoded"}

ret = requests.get(url, params = param, headers=header)
print(ret)

text = ret.json()
print(text['result']['words'])
print(text['reason'])

