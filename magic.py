import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


class Deck:

    def __init__(self,DeckName,DeckDict = {}):
        self.DeckDict = DeckDict
        self.DeckName = DeckName
        self.Features = np.copy(pd.read_csv('Training_set_'+self.DeckName+'/Features.csv',sep=';'))[:,1:]
        self.DeckList = Deck.decklist(self)
        self.CardList = Deck.cardlist(self) 

    def decklist(self):
        deck_list =[]
        Cards = self.DeckDict.items()
        for Card,NumberOfCopy in Cards:
            for _ in range(NumberOfCopy):
                deck_list.append(str(Card))
        return deck_list

    def cardlist(self):
        card_list = list(np.copy(pd.read_csv('Training_set_'+self.DeckName+'/Features.csv',sep=';'))[:,0])
        return card_list

class ML:

    def __init__(self,DeckName):
        self.DeckName = DeckName
    
    def LoadModel(self):
        self.ModelML = joblib.load('Training_set_'+self.DeckName+'/'+self.DeckName+'SavedWeights.pkl')

class Main:

    def __init__(self,DeckName,DeckDict={}):
        CurrentDeck = Deck(DeckName,DeckDict)
        CurrentDeckMLModel = ML(DeckName)
        CurrentDeckMLModel.LoadModel()
        self.DeckDict = CurrentDeck.DeckDict
        self.DeckList = CurrentDeck.DeckList
        self.CardList = CurrentDeck.CardList
        self.Features = CurrentDeck.Features
        self.DeckName = CurrentDeck.DeckName
        self.ModelML = CurrentDeckMLModel.ModelML


    def CardsToIndex(self,DeckList):
        card_index = []
        deck_list_set = list(set(DeckList))
        for card in DeckList:
            card_index.append(deck_list_set.index(card))
        return card_index,deck_list_set

    def CreateHand(self,DeckList,n):
        deck_numbers_index,deck_list_map = Main.CardsToIndex(self,DeckList) 
        deck_numbers_shuffled = [i for i in deck_numbers_index]
        np.random.shuffle(deck_numbers_shuffled)
        hand_numbers = deck_numbers_shuffled[0:n]
        hand_names = [deck_list_map[i] for i in hand_numbers]
        return hand_names  

    def CreatedHandToTestableHand(self,created_hand):
        """
        takes the output of CreateHand function as an input : turns ['card1','card2', ...] into ['card1 card2 ...'] 
        """
        testable_hand = ''
        for card in created_hand:
            testable_hand += card + ' '
        return([testable_hand])

    def SortHand(self,hand):
        hand_to_sort = [self.CardList.index(hand[i].lower()) for i in range(len(hand))]
        hand_to_sort.sort()
        hand_sorted = [self.CardList[i] for i in hand_to_sort]
        return (hand_sorted)

    def ShowHand(self,testable_hand):
        hand = testable_hand[0]
        list_im =[]
        for word in hand.split():
            list_im.append('Pictures_'+self.DeckName+'/'+word.lower()+'.jpg')
        imgs    = [ Image.open(i) for i in list_im ]
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
        imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
        plt.figure(figsize=(75,75), dpi= 50)
        plt.imshow(imgs_comb)
        plt.axis('off')
        plt.show()  

    def ConvertCardIntoFeatures(self,card):
        index = self.CardList.index(card)
        feature = self.Features[index]
        return(feature)

    def TestableHandForRFC(self,testable_hand):
        testable_hand = Main.SortHand(self,testable_hand[0].split())
        testable_hand_RFC = []
        for i in range(len(testable_hand)):
            converted_card = Main.ConvertCardIntoFeatures(self,testable_hand[i])
            testable_hand_RFC = np.concatenate((testable_hand_RFC,converted_card))
        return([testable_hand_RFC])

    def displayPrediction(self,prediction):
        if prediction == 1:
            img =  Image.open('Pictures/Keep.PNG')
            plt.figure(figsize=(75,75), dpi= 40)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

        if prediction == 0:
            img =  Image.open('Pictures/Mulligan.PNG')
            plt.figure(figsize=(75,75), dpi= 40)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

    def TestHand(self,testable_hand): 
        testable_hand_RFC = Main.TestableHandForRFC(self,testable_hand)
        prediction = self.ModelML.predict(testable_hand_RFC)[0]
        Main.displayPrediction(self,prediction)
        return prediction

    def RunHand(self,hand):
        sorted_hand = Main.SortHand(self,hand[0].split())
        testable_hand = Main.CreatedHandToTestableHand(self,sorted_hand)
        Main.ShowHand(self,testable_hand)
        Main.TestHand(self,testable_hand)

    def TestModel(self,n,DictList=[],nList=[7]):
        if DictList == []:
            DictList = [self.DeckDict]
        len_DictList = len(DictList)
        len_nList = len(nList)
        if len(self.DeckList) ==0:
            print('Error, DeckList is empty, perhaps DeckDict has been forgotten ?')
            return
        if len_DictList == 0:
            print('Error, DictList is empty, you need to give to this method a List of dictionnaries !')
            return 
        if len_nList == 0:
            print('Error, nList is empty, you need to give to this method a List of numbers of cards !')
            return
        if len_DictList !=  len_nList :
            print('Error, There is too much Dictionnaries in DictList or to much numbers in nList !')
            return
        if np.sum(nList)!=7:
            print('Error, The sum of nList is not equal to n')
            return

        ListOfDeckLists = []
        for k in range(len_nList):
            Deck_k = Deck(self.DeckName,DictList[k])
            ListOfDeckLists.append(Deck_k.DeckList)  
        
        for _ in range(n):  
            created_hand = []
            for k in range(len_nList):
                created_hand += Main.CreateHand(self,ListOfDeckLists[k],nList[k])   
            testable_hand = Main.CreatedHandToTestableHand(self,created_hand)
            Main.RunHand(self,testable_hand)

class Train:

    def __init__(self,DeckName,DeckDict,NonLandDict={},LandDict={}):

        CurrentDeck = Deck(DeckName,DeckDict)
        CurrentDeckNonLand = Deck(DeckName,NonLandDict)
        CurrentDeckLand = Deck(DeckName,LandDict)

        self.DeckName = CurrentDeck.DeckName
        self.DeckList = CurrentDeck.DeckList
        self.CardList = CurrentDeck.CardList
        self.Features = CurrentDeck.Features
        self.NonLandList = CurrentDeckNonLand.DeckList
        self.LandList = CurrentDeckLand.DeckList

    def MakeTrainingSet(self,n,TrainingSetSize,TrainingFileName):
        pv =';'
        with open('Training_set_'+self.DeckName+'/'+TrainingFileName+'.csv' , 'a+') as TrainingFile,\
        open('Training_set_'+self.DeckName+'/'+TrainingFileName+'Question.csv' , 'a+') as QuestionFile:
            for i in range(TrainingSetSize):
                current_hand = Main.CreateHand(self,self.DeckList,n)
                sorted_hand = Main.SortHand(self,current_hand)
                testable_sorted_hand = Main.CreatedHandToTestableHand(self,sorted_hand)        
                Main.ShowHand(self,testable_sorted_hand)            
                y = input("training example : "+str(i+1)+" / "+str(TrainingSetSize)+" | Keep: 1, Mull: 0 or Not sure: 3 ?")
                line_written=''
                for card in sorted_hand:
                    line_written += card+pv
                line_written += y + '\n'
                y=int(y)
                if y==1 or y==0:
                    TrainingFile.write(line_written)
                if y==3:
                    QuestionFile.write(line_written)
                if y==9:
                    break 
        print('Written !')

    def MakeTrainingSetWithModel(self,n,TrainingSetSize,TrainingFileName):
        CurrentDeckMLModel = ML(self.DeckName)
        CurrentDeckMLModel.LoadModel()
        self.ModelML = CurrentDeckMLModel.ModelML
        pv =';'
        with open('Training_set_'+self.DeckName+'/'+TrainingFileName+'.csv' , 'a+') as TrainingFile,\
        open('Training_set_'+self.DeckName+'/'+TrainingFileName+'Question.csv' , 'a+') as QuestionFile:
            for i in range(TrainingSetSize):
                current_hand = Main.CreateHand(self,self.DeckList,n)
                sorted_hand = Main.SortHand(self,current_hand)
                testable_hand = Main.CreatedHandToTestableHand(self,sorted_hand)
                Main.ShowHand(self,testable_hand) 
                prediction = Main.TestHand(self,testable_hand)     
                y = int(input("training example : "+str(i+1)+" / "+str(TrainingSetSize)+" | Correct: 1, Not_correct: 0 or Not sure: 3 ?"))
                line_written=''
                for card in sorted_hand:
                    line_written += card+pv
                if y==1:
                    prediction=str(prediction)
                    line_written += prediction + '\n'
                    TrainingFile.write(line_written)
                if y==0:
                    prediction=str(1-prediction)
                    line_written += prediction + '\n'
                    TrainingFile.write(line_written)
                if y==3:
                    y=str(y)
                    line_written += y + '\n'
                    QuestionFile.write(line_written)
                if y==9:
                    break 
        print('Written !')


    def MakeControlledTrainingSetWithModel(self,n,DictList,nList,TrainingSetSize,TrainingFileName):

        """
        - n represents the number of cards you want in your hand.
        - DictList represents the list of python dictionnaries you want to create your hand with.
        - nList represents the list of number of cards (integers) you want to pick from each dictionnary in DictList, respectively.

        """
        len_DictList = len(DictList)
        len_nList = len(nList)
        if len_DictList == 0:
            print('Error, DictList is empty, you need to give to this method a List of dictionnaries !')
            return 
        if len_nList == 0:
            print('Error, nList is empty, you need to give to this method a List of numbers of cards !')
            return
        if len_DictList !=  len_nList :
            print('Error, There is too much Dictionnaries in DictList or too much numbers in nList !')
            return
        if np.sum(nList)!=n:
            print('Error, The sum of nList is not equal to n')
            return 

        ListOfDeckLists = []
        for k in range(len_nList):
            Deck_k = Deck(self.DeckName,DictList[k])
            ListOfDeckLists.append(Deck_k.DeckList)     

        CurrentDeckMLModel = ML(self.DeckName)
        CurrentDeckMLModel.LoadModel()
        self.ModelML = CurrentDeckMLModel.ModelML

        pv =';'
        with open('Training_set_'+self.DeckName+'/'+TrainingFileName+'.csv' , 'a+') as TrainingFile,\
        open('Training_set_'+self.DeckName+'/'+TrainingFileName+'Question.csv' , 'a+') as QuestionFile:
            
            for i in range(TrainingSetSize):
                current_hand = []
                for k in range(len_nList):
                    current_hand += Main.CreateHand(self,ListOfDeckLists[k],nList[k])   
                
                sorted_hand = Main.SortHand(self,current_hand)
                testable_hand = Main.CreatedHandToTestableHand(self,sorted_hand)
                Main.ShowHand(self,testable_hand) 
                prediction = Main.TestHand(self,testable_hand)              
                y = int(input("training example : "+str(i+1)+" / "+str(TrainingSetSize)+" | Correct: 1, Not_correct: 0 or Not sure: 3 ?"))
                line_written=''
                for card in sorted_hand:
                    line_written += card+pv
                if y==1:
                    prediction=str(prediction)
                    line_written += prediction + '\n'
                    TrainingFile.write(line_written)
                if y==0:
                    prediction=str(1-prediction)
                    line_written += prediction + '\n'
                    TrainingFile.write(line_written)
                if y==3:
                    y=str(y)
                    line_written += y + '\n'
                    QuestionFile.write(line_written)
                if y==9:
                    break 
        print('Written !')

    def TrainingSetToDocs(self,TrainingFileName):
        training_set = pd.read_csv('Training_set_'+self.DeckName+'/'+TrainingFileName+'.csv',sep =';',header=None)
        training_set = np.copy(training_set)
        docs = []
        for line in training_set:
            line = line[0:7]
            docs.append(" ".join(line))
        labels = training_set[:,7]           
        return(docs,labels)

    def WriteTraininsSetFeatureFromDocs(self,docs,labels,TrainingFileName):
        pv =';'
        with open('Training_set_'+self.DeckName+'/'+TrainingFileName+'.csv' , 'w') as TrainingFile:
            for i in range(len(docs)):
                docs_i = Main.SortHand(self,docs[i].split())
                line_to_write=''
                for j in range(7):
                    feature_card = Main.ConvertCardIntoFeatures(self,docs_i[j])
                    K = len(feature_card)
                    for k in range(K):
                        line_to_write += str(int(feature_card[k]))+pv
                line_to_write+=str(labels[i])
                TrainingFile.write(line_to_write+'\n')
        print('Done !')

    def TransformTrainingSet(self,TrainingFileName):
        docs,labels = Train.TrainingSetToDocs(self,TrainingFileName)
        Train.WriteTraininsSetFeatureFromDocs(self,docs,labels,'TrainingSet')

    def TrainAndSaveWeights(self,save=True):
        data = np.copy(pd.read_csv('Training_set_'+self.DeckName+'/TrainingSet.csv',sep=';',header=None))
        X, y = data[:,:-1],data[:,-1]
        print("N_examples : ",X.shape[0])
        MLModel = RandomForestClassifier(n_estimators=100, random_state=0)
        MLModel.fit(X, y)
        print(MLModel.score(X,y))
        self.ModelML = MLModel
        if save:
            joblib.dump(MLModel,'Training_set_'+self.DeckName+'/'+self.DeckName+'SavedWeights.pkl')
    
    def TrainAndTest(self,TestSize):
        data = np.copy(pd.read_csv('Training_set_'+self.DeckName+'/TrainingSet.csv',sep=';',header=None))
        X, y = data[:,:-1],data[:,-1]
        print("N_examples : ",X.shape[0])
        MLModel = RandomForestClassifier(n_estimators=100, random_state=0)
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=TestSize)
        print("N_training examples : ",X_train.shape[0])
        print("N_test examples : ",X_test.shape[0])
        MLModel.fit(X_train,y_train)
        print(MLModel.score(X_train,y_train))
        print(MLModel.score(X_test,y_test))