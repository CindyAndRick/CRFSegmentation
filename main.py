from CRF import CRF

def shell():
    crf = CRF(templatePath="dataset/dataset2/template.utf8", scoreMapPath="model\scoreMap", trainDataPath="dataset/dataset2/train.utf8")
    while True:
        sentence = input("请输入句子：")
        if sentence == "exit":
            break
        print(crf.predict(sentence))

if __name__ == '__main__':
    # crf.start_train(100)
    shell()