"""
    基于FastAPI的后端纠错API接口服务
    先加载文本纠错模型预热再启动后端接口服务
"""
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sutil import cut_sent, replace_char, get_paragraphs_text
import uvicorn
from paddlenlp import Taskflow
import time
from paddlenlp.transformers import ErnieTokenizer
from paddlenlp.data import Vocab
from predict import Predictor
import requests

# 2.加载训练好的文本纠错模型去完成纠错任务
tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
pinyin_vocab = Vocab.load_vocabulary("./best_model/pinyin_vocab.txt", unk_token='[UNK]', pad_token='[PAD]')
# 配置加载模型参数地址
predictor = Predictor('./best_model/static_graph_params.pdmodel', './best_model/static_graph_params.pdiparams', 'cpu', 128, tokenizer, pinyin_vocab)

print("模型加载预热！")
toCorrectText = [
        '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
        '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。',
    ]

# PaddleNLP 文本纠错
# text_correction = Taskflow("text_correction")
# PaddleNLP 文本摘要
text_summarization = Taskflow("text_summarization")

# 纠错结果处理
print("PaddleNLP文本纠错结果：")
for idx, item in enumerate(toCorrectText):
    res = predictor.predict(item, batch_size=2)
    print(res)

# 创建一个 FastAPI「实例」，名字为app
app = FastAPI()

# 设置允许跨域请求，解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求体数据类型：text
class Document(BaseModel):
    text: str


# 定义路径操作装饰器：post方法 + API接口路径
@app.post("/v1/antonym/", status_code=200)
# 定义路径操作函数，当接口被访问将调用该函数
async def TextSummarization(document: Document):
    try:
        text = document.text
        url = 'http://apis.juhe.cn/tyfy/query'
        param = {"word": text, "type": "2", "key": "a9f946f996eadf163b94c33f53512687"}
        header = {"Content-Type": "application/x-www-form-urlencoded"}
        ret = requests.get(url, params = param, headers=header)
        result = ret.json()
        # 接口结果返回
        results = {"message": result['reason'], "originalText": document.text, "correctionResults": result['result']['words']}
        return results
    # 异常处理
    except Exception as e:
        print("异常信息：", e)
        raise HTTPException(status_code=500, detail=str("请求失败，服务器端发生异常！异常信息提示：" + str(e)))
@app.post("/v1/synonym/", status_code=200)
# 定义路径操作函数，当接口被访问将调用该函数
async def TextSummarization(document: Document):
    try:
        text = document.text
        url = 'http://apis.juhe.cn/tyfy/query'
        param = {"word": text, "key": "a9f946f996eadf163b94c33f53512687"}
        header = {"Content-Type": "application/x-www-form-urlencoded"}
        ret = requests.get(url, params = param, headers=header)
        result = ret.json()
        # 接口结果返回
        results = {"message": result['reason'], "originalText": document.text, "correctionResults": result['result']['words']}
        return results
    # 异常处理
    except Exception as e:
        print("异常信息：", e)
        raise HTTPException(status_code=500, detail=str("请求失败，服务器端发生异常！异常信息提示：" + str(e)))
@app.post("/v1/summarization/", status_code=200)
# 定义路径操作函数，当接口被访问将调用该函数
async def TextSummarization(document: Document):
    try:
        # 获取要进行摘要的文本内容
        text = document.text
        # 文本摘要
        result = text_summarization(text)

        # 接口结果返回
        results = {"message": "success", "originalText": document.text, "correctionResults": result}
        return results
    # 异常处理
    except Exception as e:
        print("异常信息：", e)
        raise HTTPException(status_code=500, detail=str("请求失败，服务器端发生异常！异常信息提示：" + str(e)))
# 文本纠错接口
@app.post("/v1/textCorrect/", status_code=200)
# 定义路径操作函数，当接口被访问将调用该函数
async def TextErrorCorrection(document: Document):
    try:
        # 获取要进行纠错的文本内容
        text = document.text
        # 精细分句处理以更好处理长文本
        data = cut_sent(text)
        result = predictor.predict(data, batch_size=2)
        # 拼接分句后结果
        correctionResult = ''
        for temp in result:
            if temp is not '':
                correctionResult += temp;
                correctionResult += '\n';
        # correctionResult = "\n".join(result)
        # 接口结果返回
        results = {"message": "success", "originalText": document.text, "correctionResults": correctionResult}
        return results
    # 异常处理
    except Exception as e:
        print("异常信息：", e)
        raise HTTPException(status_code=500, detail=str("请求失败，服务器端发生异常！异常信息提示：" + str(e)))

# 文档纠错接口
@app.post("/v1/docCorrect/", status_code=200)
# 定义路径操作函数，当接口被访问将调用该函数
async def DocumentErrorCorrection(file: UploadFile):
    # 读取上传的文件
    docBytes = file.file.read()
    docName = file.filename
    # 判断上传文件类型
    docType = docName.split(".")[-1]
    if docType != "doc" and docType != "docx":
        raise HTTPException(status_code=406, detail=str("请求失败，上传文档格式不正确！请上传word文档！"))
    try:
        # 将上传文件保存到本地，添加时间标记避免重复
        now_time = int(time.mktime(time.localtime(time.time())))
        docPath = "./fileres/" + str(now_time) + "_" + docName
        fout = open(docPath, 'wb')
        fout.write(docBytes)
        fout.close()

        # 读取要进行文本纠错的word文档内容
        docText = get_paragraphs_text(docPath)
        # 对word文档内容进行分句处理避免句子过长
        docText = cut_sent(docText)

        # 进行文本纠错和标记
        correctionResult = ""
        for idx, item in enumerate(docText):
            if item is not '':
                res = predictor.predict(item, batch_size=2)
                length = len(res[0]['errors'])
                if length > 0:
                    for i, error in enumerate(res[0]['errors']):
                        if i == 0:
                            item = replace_char(item, (list(res[0]['errors'][i]['correction'].keys())[0] + '（' + list(res[0]['errors'][i]['correction'].values())[0] + '）'), res[0]['errors'][i]['position'])
                        else:
                            # 如果句子中有多处错字，那么每替换前面一个字，后面的错字索引往后移动3位：即括号+字=3位
                            p = res[0]['errors'][i]['position'] + i * 3
                            item = replace_char(item, (list(res[0]['errors'][i]['correction'].keys())[0] + '（' + list(res[0]['errors'][i]['correction'].values())[0] + '）'), p)
                if item is not '':
                    correctionResult += item;
                    correctionResult += '\n';

        # 接口结果返回
        results = {"message": "success", "docText": str(docText), "correctionResults": correctionResult}
        return results
    # 异常处理
    except Exception as e:
        print("异常信息：", e)
        raise HTTPException(status_code=500, detail=str("请求失败，服务器端发生异常！异常信息提示：" + str(e)))



# 启动创建的实例app，设置启动ip和端口号
uvicorn.run(app, host="127.0.0.1", port=8000)
