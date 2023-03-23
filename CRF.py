import pickle
from datetime import datetime

class CRF:
    scoreMap = dict()
    template = list()
    scoreMapPath = ""
    templatePath = ""
    trainDataPath = ""
    rowStatus = {0: "B", 1: "M", 2: "E", 3: "S"}
    statusRow = {"B": 0, "M": 1, "E": 2, "S": 3}

    def __init__(self, templatePath, scoreMapPath, trainDataPath):
        self.scoreMap = self.load_obj(scoreMapPath)
        self.template = self.readTemplate(templatePath)
        self.scoreMapPath = scoreMapPath
        self.templatePath = templatePath
        self.trainDataPath = trainDataPath

    def getStrBtw(self, s, b, e):
        a = list()
        index1 = 0
        index2 = 0
        for x in range(len(s)):
            if s[x] == b:
                index1 = x
                continue
            if s[x] == e:
                index2 = x
                a.append(int(s[index1+1:index2]))
        return a

    def readTemplate(self, fileName):
        gram = list()
        tp = open(fileName, encoding='UTF-8')
        unigram = list()
        bigram = list()
        for line in tp:
            if len(line)>1:
                term = self.getStrBtw(line,"[",",")
                if (line[0] == "U")and(len(term)>0):
                     unigram.append(term)
                if (line[0] == "B")and(len(term)>0):
                     bigram.append(term)
        gram.append(unigram)
        gram.append(bigram)
        return gram

    def getUniTemplate(self):
        return self.template[0]
    
    def getBiTemplate(self):
        return self.template[1]
    
    def segment(self, sentence):
        sentenceLen = len(sentence)
        statusFrom = [["" for _ in range(sentenceLen)] for _ in range(4)]
        maxScore = [[-1 for _ in range(sentenceLen)] for _ in range(4)]
        for col in range(sentenceLen):
            for row in range(4):
                curStatus = self.rowStatus[row]
                if col == 0:
                    uniScore = self.getUniScore(sentence, 0, curStatus)
                    biScore = self.getBiScore(sentence, 0, " ", curStatus)
                    maxScore[row][0] = uniScore + biScore
                else:
                    scores = [-1 for _ in range(4)]
                    for i in range(4):
                        preStatus = self.rowStatus[i]
                        transScore = maxScore[i][col - 1]
                        uniScore = self.getUniScore(sentence, col, curStatus)
                        biScore = self.getBiScore(sentence, col, preStatus, curStatus)
                        scores[i] = transScore + uniScore + biScore
                    maxIndex = scores.index(max(scores))
                    maxScore[row][col] = scores[maxIndex]
                    statusFrom[row][col] = self.rowStatus[maxIndex]
        resBuf = ['' for _ in range(sentenceLen)]
        scoreBuf = [0 for _ in range(4)]
        for i in range(4):
            scoreBuf[i] = maxScore[i][sentenceLen - 1]
        resBuf[sentenceLen - 1] = self.rowStatus[scoreBuf.index(max(scoreBuf))]
        for backIndex in range(sentenceLen - 2, -1, -1):
            resBuf[backIndex] = statusFrom[self.statusRow[resBuf[backIndex + 1]]][backIndex + 1]
        res = ""
        for i in range(sentenceLen):
            res += resBuf[i]
        return res

    def getUniScore(self, sentence, curPos, curStatus):
        uniScore = 0
        uniTemplate = self.getUniTemplate()
        num = len(uniTemplate)
        for i in range(num):
            key = self.makeKey(uniTemplate[i], str(i), sentence, curPos, curStatus)
            if key in self.scoreMap.keys():
                uniScore += self.scoreMap[key]
        return uniScore
    
    def getBiScore(self, sentence, curPos, preStatus, curStatus):
        biScore = 0
        biTemplate = self.getBiTemplate()
        num = len(biTemplate)
        for i in range(num):
            key = self.makeKey(biTemplate[i], str(i), sentence, curPos, preStatus + curStatus)
            if key in self.scoreMap.keys():
                biScore += self.scoreMap[key]
        return biScore
    
    def makeKey(self, template, identity, sentence, pos, statusCovered):
        str = ""
        str += identity
        for offset in template:
            index = pos + offset
            if index < 0 or index >= len(sentence):
                str += " "
            else:
                str += sentence[index]
        str += "/"
        str += statusCovered
        return str

    def train(self, sentence, truth, save=True):
        prediction = self.segment(sentence)
        wrongNum = 0
        for i in range(len(sentence)):
            if prediction[i] != truth[i]:
                wrongNum += 1
                self.updateWeights("U", sentence, i, truth, prediction)
                self.updateWeights("B", sentence, i, truth, prediction)
        return wrongNum

    def updateWeights(self, type, sentence, index, truth, prediction):
        template = []
        num = 0
        if type == "U":
            template = self.getUniTemplate()
            num = len(template)
            for i in range(num):
                predictKey = self.makeKey(template[i], str(i), sentence, index, prediction[index])
                if predictKey in self.scoreMap.keys():
                    self.scoreMap[predictKey] -= 1
                else:
                    self.scoreMap[predictKey] = -1
                turthKey = self.makeKey(template[i], str(i), sentence, index, truth[index])
                if turthKey in self.scoreMap.keys():
                    self.scoreMap[turthKey] += 1
                else:
                    self.scoreMap[turthKey] = 1
                # print("uni", predictKey, turthKey)
        if type == "B":
            template = self.getBiTemplate()
            num = len(template)
            for i in range(num):
                predictKey = self.makeKey(template[i], str(i), sentence, index, prediction[index] if index < 1 else prediction[index - 1 : index + 1])
                if predictKey in self.scoreMap.keys():
                    self.scoreMap[predictKey] -= 1
                else:
                    self.scoreMap[predictKey] = -1
                truthKey = self.makeKey(template[i], str(i), sentence, index, truth[index] if index < 1 else truth[index - 1 : index + 1])
                if truthKey in self.scoreMap.keys():
                    self.scoreMap[truthKey] += 1
                else:
                    self.scoreMap[truthKey] = 1
                # print("bi", predictKey, truthKey)
            
    def start_train(self, iter, save=True):
        data = self.preprocessData()
        sentences = data[0]
        tags = data[1]
        for it in range(iter):
            wrong = 0
            total = 0
            for i in range(len(sentences)):
                if(len(sentences[i])>0):
                    wrong += self.train(sentences[i], tags[i])
                    total += len(sentences[i])
                    if i % 3000 == 0:
                        print("iter:" + str(it) + " " + "{:5}".format(i) + "/" + str(len(sentences)) +" acc:" + str((total - wrong) / total))
            if save:
                self.save_obj(self.scoreMap, self.scoreMapPath, it)

    def predict(self, sentence):
        temp = self.segment(sentence)
        res = ""
        for i in range(len(temp)):
            res += sentence[i]
            if temp[i]=='E' or temp[i]=='S':
                res+=' '         
        return res

    def preprocessData(self):
        print("preparing data...")
        ifp = open(self.trainDataPath, encoding='UTF-8')
        sentence_set = list()
        tag_set = list()
        sentence = ""
        tag = ""
        for line in ifp:
            if (len(line)<4):
                sentence_set.append(sentence)
                tag_set.append(tag)
                sentence=""
                tag=""
            else:
                line.split()
                sentence+=line[0]
                tag+=line[2]
        train_data = [sentence_set, tag_set]
        print("data prepared")
        return train_data

    def save_obj(self, obj, name, it):
        with open(name + '-' + str(it) + 'iter' + str(datetime.now()).replace(':', '-') + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)