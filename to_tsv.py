import pandas as pd
file_path = "data/data122751/train.csv"
classify_rawtext = pd.read_csv(file_path, sep=",",usecols=['class','text'])
ner_text = pd.read_csv(file_path, sep=",",usecols=['text','BIO_anno'])
classify_rawtext = classify_rawtext.sample(frac=1)  # 打乱数据集
ner_text = ner_text.sample(frac=1)  # 打乱数据集

print(len(classify_rawtext))
print(len(ner_text))

gen_classify=True
gen_ner=True
gen_final=True

if gen_classify:
    cols = list(classify_rawtext)
    # print(cols)
    cols.insert(0, cols.pop(cols.index('class')))
    # print(cols)
    classify_text = classify_rawtext.loc[:, cols]

    classify_train = classify_text[:int(len(classify_text) * 0.95)]
    classify_dev = classify_text[int(len(classify_text) * 0.95):] #classify_text[int(len(classify_text) * 0.8):int(len(classify_text) * 0.9)]
    classify_test = classify_text[int(len(classify_text) * 0.95):]
    classify_train.to_csv('data/data122751/classify_train_data.tsv', sep='\t', header=None, index=False, columns=None, mode="w")
    classify_dev.to_csv('data/data122751/classify_dev_data.tsv', sep='\t', header=None, index=False, columns=None, mode="w")
    classify_test.to_csv('data/data122751/classify_test_data.tsv', sep='\t', header=None, index=False, columns=None, mode="w")

    # 验证train,dev标签分布是否均匀
    for file in ['classify_train_data', 'classify_dev_data','classify_test_data']:
        file_path = f"data/data122751/{file}.tsv"
        text = pd.read_csv(file_path, sep="\t", header=None)
        prob = dict()
        total = len(text[0])
        for i in text[0]:
            if prob.get(i) is None:
                prob[i] = 1
            else:
                prob[i] += 1
        # 按标签排序
        prob = {i[0]: round(i[1] / total, 3) for i in sorted(prob.items(), key=lambda k: k[0])}
        print(file, prob, total)

split_char = "\002"
if gen_ner:
    sentence_cnt=len(ner_text)
    for i in range(sentence_cnt):
        raw_sentence=ner_text["text"][i].lower()
        raw_sentence=split_char.join(raw_sentence)
        ner_text.loc[i,"text"]=raw_sentence

        raw_lable=ner_text["BIO_anno"][i]
        raw_lable=raw_lable.replace(" ",split_char)
        ner_text.loc[i,"BIO_anno"]=raw_lable

    ner_train = ner_text[:int(len(ner_text) * 0.95)]
    ner_dev = ner_text[int(len(ner_text) * 0.95):] #ner_text[int(len(ner_text) * 0.8):int(len(ner_text) * 0.9)]
    ner_test = ner_text[int(len(ner_text) * 0.95):]

    ner_train.to_csv('data/data122751/ner_train_data.tsv', sep='\t', header=None, index=False, columns=None, mode="w")
    ner_dev.to_csv('data/data122751/ner_dev_data.tsv', sep='\t', header=None, index=False, columns=None, mode="w")
    ner_test.to_csv('data/data122751/ner_test_data.tsv', sep='\t', header=None, index=False, columns=None, mode="w")

if gen_final:
    finaltest = pd.read_csv("data/data122751/test.csv", sep=",")
    finaltest.to_csv('data/data122751/classify_finaltest_data.tsv', sep='\t', header=None, index=False, columns=None, mode="w")
    for i in range(len(finaltest)):
        raw_sentence=finaltest["text"][i]
        raw_sentence=split_char.join(raw_sentence)
        finaltest.loc[i,"text"]=raw_sentence
    finaltest.to_csv('data/data122751/ner_finaltest_data.tsv', sep='\t', header=None, index=False, columns=None, mode="w")


