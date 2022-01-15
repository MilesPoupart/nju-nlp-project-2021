import paddlehub as hub
import pandas as pd

if __name__ == '__main__':
    #file_path = "data/data122751/ner_test_data.tsv"
    file_path = "data/data122751/ner_finaltest_data.tsv"
    text = pd.read_csv(file_path, sep="\t", header=None)
    label_list=['B-BANK', 'I-BANK', 'B-PRODUCT', 'I-PRODUCT', 'B-COMMENTS_N', 'I-COMMENTS_N', 'B-COMMENTS_ADJ', 'I-COMMENTS_ADJ', 'O']
    label_map = {idx: label for idx, label in enumerate(label_list)}
    data = [[i] for i in text[1]]

    model = hub.Module(name='ernie', task='token-cls',
                        load_checkpoint='./ernie_checkpoint_ner/best_model/model.pdparams', label_map=label_map)
    results = model.predict(data, max_seq_len=128, batch_size=1, use_gpu=True)

    # print(data[0])
    # print(results[0])

    resultlist=[]
    for idx, text in enumerate(data):
        resultlist.append(" ".join(results[idx][1:len(text[0])+1]))
        # if idx<10:
        #     labels = results[idx][1:len(text[0])+1]        
        #     print(f'Data: {text} \t Label: {", ".join(results[idx][1:len(text[0])+1])}')

    resultsDF = pd.DataFrame(data=resultlist)
    resultsDF.to_csv('ner_result.csv',sep='\t', header=None, index=False, columns=None, mode="w")

    # 输出测试集准确率
    # count = 0
    # for i, j in zip(text[0], results):
    #     # print(type(i), type(j))
    #     if int(i) == int(j):
    #         count += 1
    # print("测试集准确率:", count / len(results))
