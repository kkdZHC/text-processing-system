"""
    demo简易文本纠错演示程序
    PaddleNLP一键预测工具Taskflow的快捷文本纠错体验
    加载已经训练好的文本纠错模型进行文本纠错
"""

from paddlenlp import Taskflow
from paddlenlp.transformers import ErnieTokenizer
from paddlenlp.data import Vocab
from predict import Predictor

if __name__ == "__main__":
    # 1.Taskflow一键体验文本纠错功能
    text_summarization = Taskflow("text_summarization")
    text_correction = Taskflow("text_correction")
    # 要进行纠错的文本
    samples1 = [
        '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
        '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。',
    ]
    sample2 = '5月22日，三穗县2023年专职消防队伍执勤岗位大练兵比武竞赛在三穗县消防救援大队举行。来自该县8支政府专职消防队伍同台竞技、各展风采，参赛队员敢于超越、奋勇争先，充分展示了过硬的专业技能、顽强的竞技作风和良好的精神风貌。'
    # 文本纠错
    result1 = text_summarization(sample2)
    result2 = text_correction(samples1)
    print(result1)
    #输出纠错结果
    for source, target in zip(samples1, result2):
        print("要纠错的文本为:\n", source)
        print("纠错后的文本为:\n", target['target'])
        print("错误位置为：\n", target['errors'])
        print('\n')

    # 2.加载训练好的文本纠错模型去完成纠错任务
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    pinyin_vocab = Vocab.load_vocabulary("./best_model/pinyin_vocab.txt", unk_token='[UNK]', pad_token='[PAD]')
    # 配置加载模型参数地址
    predictor = Predictor('./best_model/static_graph_params.pdmodel', './best_model/static_graph_params.pdiparams', 
                          'cpu', 128, tokenizer, pinyin_vocab)

    # 要进行纠错的文本
    samples2 = [
        '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
        '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。',
    ]

    # 文本纠错
    results2 = predictor.predict(samples2, batch_size=2)
    # 输出纠错结果
    for source, target in zip(samples2, results2):
        print("要纠错的文本为:", source)
        print("纠错后的文本为:", target)
        print('\n')



