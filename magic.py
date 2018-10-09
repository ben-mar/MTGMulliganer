import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unittest
from PIL import Image
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


NUMERIC_FEATURE_SPLIT = 1
NAME_CARDS_INDEX = 0
PREDICTION_INDEX = 0
NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN = 7

DPI_DISPLAY_PREDICTION = 40
DPI_SHOW_HAND = 50
FIG_SIZE = (75,75)

class Utility:

    def __init__(self):
        None

    def read(self,path,header=0):
        return np.copy(pd.read_csv(path,sep=';',header=header))
    
    def _DisplayImage(self,Image,FigSize,Dpi):
        plt.figure(figsize=FigSize, dpi= Dpi)
        plt.imshow(Image)
        plt.axis('off')
        plt.show() 

class Deck:

    def __init__(self,DeckName,DeckDict = {}):
        self.DeckDict = DeckDict
        self.DeckName = DeckName
        self.Features = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/Features.csv')[:,NUMERIC_FEATURE_SPLIT:]
        self.DeckList = Deck._decklist(self)
        self.CardList = Deck._cardlist(self) 

    def _decklist(self):
        DeckList = []
        Cards = self.DeckDict.items()
        for Card,NumberOfCopy in Cards:
            for _ in range(NumberOfCopy):
                DeckList.append(Card)
        return DeckList

    def _cardlist(self):
        CardList = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/Features.csv')[:,NAME_CARDS_INDEX]
        CardList = list(CardList)
        return CardList

class ML:

    def __init__(self,DeckName):
        self.DeckName = DeckName
    
    def LoadModel(self):
        self.ModelML = joblib.load(self.DeckName+'/Training_set_'+self.DeckName+'/'+self.DeckName+'SavedWeights.pkl')

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


    def _CardsToIndex(self,DeckList):
        DeckListNumbers = []
        for Card in DeckList:
            DeckListNumbers.append(self.CardList.index(Card))
        return DeckListNumbers

    def _CreateHand(self,DeckList,n):
        """
        Creates hand from the DeckList with n cards in it, which is a list of n strings containing the names of the card.
        This kind of structure will be called HandNames from now on.
        """

        if n>len(DeckList):
            print("Error, n > len(DeckList), ({0} > {2}) the function CreateHand cannot create a hand with {0} cards from a DeckList"
            " only composed by the following cards {1}".format(n,DeckList,len(DeckList)))
            return          

        DeckListNumbers = Main._CardsToIndex(self,DeckList) 
        DeckListNumbersShuffled = [i for i in DeckListNumbers]
        np.random.shuffle(DeckListNumbersShuffled)
        HandNumbers = DeckListNumbersShuffled[0:n]
        HandNames = [self.CardList[i] for i in HandNumbers]
        return HandNames 

    def CreateHandFromDicts(self,DictList=[],nList=[NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN]):
        """
        - n represents the number of cards you want in your hand.
        - DictList represents the list of python dictionnaries you want to create your hand with.
        - nList represents the list of number of cards (integers) you want to pick from each dictionnary in DictList, respectively.

        """

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
        if np.sum(nList)!=NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN:
            print('Error, The sum of nList is not equal to the numbers of cards without mulligan, which is 7 !')
            return

        ListOfDeckLists = []
        for k in range(len_nList):
            Deck_k = Deck(self.DeckName,DictList[k])
            ListOfDeckLists.append(Deck_k.DeckList)

        HandNames = []
        for k in range(len_nList):
            HandNames += Main._CreateHand(self,ListOfDeckLists[k],nList[k])
        return HandNames


    def SortHand(self,HandNames):
        """
        Sorts a hand that is a list of str (the different cards in it),
        according to the list defined in the features.csv file.
        It return an output which is the sorted list of str.
        """

        HandToSortNumbers = [self.CardList.index(HandNames[i]) for i in range(len(HandNames))]
        HandToSortNumbers.sort()
        SortedHandNames = [self.CardList[i] for i in HandToSortNumbers]
        return (SortedHandNames)

    def _ConvertCardIntoFeatures(self,Card):
        """
        Takes a card as an input and returns a 1D array corresponding to the features of the card.
        """
        CardIndex = self.CardList.index(Card)
        CardFeature = self.Features[CardIndex]
        return(CardFeature)
    
    def _MakeTestableHand(self,HandNames):
        """
        takes a hand that is a list of str (the different cards in it) as an input and returns a list of one numpy 1D-array representing 
        the features of the Cards, making the output ready for the Scikit-Learn.predict() function
        """
        TestableHand = []
        for i in range(len(HandNames)):
            Card_i = HandNames[i]
            CardFeature_i = Main._ConvertCardIntoFeatures(self,Card_i)
            TestableHand = np.concatenate((TestableHand,CardFeature_i))
        TestableHand = [TestableHand]
        return(TestableHand)

    def ShowHand(self,HandNames):
        ImagesList =[]
        for Card in HandNames:
            ImagesList.append(self.DeckName+'/Pictures_'+self.DeckName+'/'+Card+'.jpg')
        Images = [ Image.open(i) for i in ImagesList ]
        MinimumShape = sorted( [(np.sum(i.size), i.size ) for i in Images])[0][1]
        ImagesCombined = np.hstack( (np.asarray( i.resize(MinimumShape) ) for i in Images ) )
        Utility._DisplayImage(self,ImagesCombined,FIG_SIZE,DPI_SHOW_HAND)
        

    def _displayPrediction(self,prediction):
        if prediction == 1:
            Img =  Image.open('General/Pictures/Keep.PNG')
            Utility._DisplayImage(self,Img,FIG_SIZE,DPI_DISPLAY_PREDICTION)

        if prediction == 0:
            Img =  Image.open('General/Pictures/Mulligan.PNG')
            Utility._DisplayImage(self,Img,FIG_SIZE,DPI_DISPLAY_PREDICTION)

    def TestHand(self,HandNames): 
        TestableHand = Main._MakeTestableHand(self,HandNames)
        prediction = self.ModelML.predict(TestableHand)[PREDICTION_INDEX]
        Main._displayPrediction(self,prediction)
        return prediction

    def RunHand(self,Hand):
        """
        takes a unique string as input which is the concatenation of all cards contained in the Hand
        """
        HandNames = Hand.split()
        SortedHandNames = Main.SortHand(self,HandNames)
        Main.ShowHand(self,SortedHandNames)
        Main.TestHand(self,SortedHandNames)

    def TestModel(self,n,DictList=[],nList=[NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN]):
        for _ in range(n):  
            HandNames = Main.CreateHandFromDicts(self,DictList=DictList,nList=nList)
            Hand = ' '.join(HandNames)
            Main.RunHand(self,Hand)

class Train:

    def __init__(self,DeckName,DeckDict):

        CurrentDeck = Deck(DeckName,DeckDict)
        self.DeckName = CurrentDeck.DeckName
        self.DeckList = CurrentDeck.DeckList
        self.CardList = CurrentDeck.CardList
        self.Features = CurrentDeck.Features

    def MakeTrainingSet(self,n,TrainingSetSize,TrainingFileName):
        pv =';'
        with open(self.DeckName+'/Training_set_'+self.DeckName+'/'+TrainingFileName+'.csv' , 'a+') as TrainingFile,\
        open(self.DeckName+'/Training_set_'+self.DeckName+'/'+TrainingFileName+'Question.csv' , 'a+') as QuestionFile:
            for i in range(TrainingSetSize):
                HandNames = Main._CreateHand(self,self.DeckList,n)
                SortedHandNames = Main.SortHand(self,HandNames)     
                Main.ShowHand(self,SortedHandNames)            
                Label = input("training example : "+str(i+1)+" / "+str(TrainingSetSize)+" | Keep: 1, Mull: 0 or Not sure: 3 ?")
                LineWritten=''
                for card in SortedHandNames:
                    LineWritten += card+pv
                LineWritten += Label + '\n'
                Label=int(Label)
                if Label==1 or Label==0:
                    TrainingFile.write(LineWritten)
                if Label==3:
                    QuestionFile.write(LineWritten)
                if Label==9:
                    break 
        print('Written !')

    def MakeTrainingSetWithModel(self,n,TrainingSetSize,TrainingFileName):
        CurrentDeckMLModel = ML(self.DeckName)
        CurrentDeckMLModel.LoadModel()
        self.ModelML = CurrentDeckMLModel.ModelML
        pv =';'
        with open(self.DeckName+'/Training_set_'+self.DeckName+'/'+TrainingFileName+'.csv' , 'a+') as TrainingFile,\
        open(self.DeckName+'/Training_set_'+self.DeckName+'/'+TrainingFileName+'Question.csv' , 'a+') as QuestionFile:
            for i in range(TrainingSetSize):
                HandNames = Main._CreateHand(self,self.DeckList,n)
                SortedHandNames = Main.SortHand(self,HandNames)
                Main.ShowHand(self,SortedHandNames)
                Prediction = Main.TestHand(self,SortedHandNames)     
                Label = int(input("training example : "+str(i+1)+" / "+str(TrainingSetSize)+" | Correct: 1, Not_correct: 0 or Not sure: 3 ?"))
                line_written=''
                for card in SortedHandNames:
                    line_written += card+pv
                if Label==1:
                    Prediction=str(Prediction)
                    line_written += Prediction + '\n'
                    TrainingFile.write(line_written)
                if Label==0:
                    Prediction=str(1-Prediction)
                    line_written += Prediction + '\n'
                    TrainingFile.write(line_written)
                if Label==3:
                    Label=str(Label)
                    line_written += Label + '\n'
                    QuestionFile.write(line_written)
                if Label==9:
                    break 
        print('Written !')


    def MakeControlledTrainingSetWithModel(self,n,DictList,nList,TrainingSetSize,TrainingFileName):

        """
        - n represents the number of cards you want in your hand.
        - DictList represents the list of python dictionnaries you want to create your hand with.
        - nList represents the list of number of cards (integers) you want to pick from each dictionnary in DictList, respectively.

        """

        CurrentDeckMLModel = ML(self.DeckName)
        CurrentDeckMLModel.LoadModel()
        self.ModelML = CurrentDeckMLModel.ModelML

        pv =';'
        with open(self.DeckName+'/Training_set_'+self.DeckName+'/'+TrainingFileName+'.csv' , 'a+') as TrainingFile,\
        open(self.DeckName+'/Training_set_'+self.DeckName+'/'+TrainingFileName+'Question.csv' , 'a+') as QuestionFile:
            
            for i in range(TrainingSetSize):
                HandNames = Main.CreateHandFromDicts(self,DictList=DictList,nList=nList)
                SortedHandNames = Main.SortHand(self,HandNames)
                Main.ShowHand(self,SortedHandNames)
                Prediction = Main.TestHand(self,SortedHandNames)              
                Label = int(input("training example : "+str(i+1)+" / "+str(TrainingSetSize)+" | Correct: 1, Not_correct: 0 or Not sure: 3 ?"))
                LineWritten=''
                for Card in SortedHandNames:
                    LineWritten += Card+pv
                if Label==1:
                    Prediction=str(Prediction)
                    LineWritten += Prediction + '\n'
                    TrainingFile.write(LineWritten)
                if Label==0:
                    Prediction=str(1-Prediction)
                    LineWritten += Prediction + '\n'
                    TrainingFile.write(LineWritten)
                if Label==3:
                    Label=str(Label)
                    LineWritten += Label + '\n'
                    QuestionFile.write(LineWritten)
                if Label==9:
                    break 
        print('Written !')

    def TrainingSetToDocs(self,TrainingFileName):
        TrainingSet = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/'+TrainingFileName+'.csv',header=None)
        Docs = []
        for Line in TrainingSet:
            Line = Line[0:NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN]
            Docs.append(" ".join(Line))
        Labels = TrainingSet[:,NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN]           
        return(Docs,Labels)


    def WriteTrainingSetFeatureFromDocs(self,Docs,Labels,TrainingFileName):
        pv =';'
        with open(self.DeckName+'/Training_set_'+self.DeckName+'/'+TrainingFileName+'.csv' , 'w') as TrainingFile:
            for i in range(len(Docs)):
                Docs_i = Main.SortHand(self,Docs[i].split())
                LineWritten=''
                for j in range(NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN):
                    CardFeature = Main._ConvertCardIntoFeatures(self,Docs_i[j])
                    K = len(CardFeature)
                    for k in range(K):
                        LineWritten += str(int(CardFeature[k]))+pv
                LineWritten+=str(Labels[i])
                TrainingFile.write(LineWritten+'\n')
        print('Written !')

    def TransformTrainingSet(self,TrainingFileInput,TrainingFileOutput= 'TrainingSet'):
        Docs,Labels = Train.TrainingSetToDocs(self,TrainingFileInput)
        Train.WriteTrainingSetFeatureFromDocs(self,Docs,Labels,TrainingFileOutput)

    def TrainAndSaveWeights(self,save=True):
        data = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/TrainingSet.csv',header = None)
        X, y = data[:,:-1],data[:,-1]
        N_examples = X.shape[0]
        print("N_examples : ",N_examples)
        MLModel = RandomForestClassifier(n_estimators=100, random_state=0)
        MLModel.fit(X, y)
        print(MLModel.score(X,y))
        self.ModelML = MLModel
        if save:
            joblib.dump(MLModel,self.DeckName+'/Training_set_'+self.DeckName+'/'+self.DeckName+'SavedWeights.pkl')
    
    def TrainAndTest(self,TestSize=0,TestingFileInput=''):
        data = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/TrainingSet.csv',header = None)
        X, y = data[:,:-1],data[:,-1]
        print("N_examples : ",X.shape[0])
        if TestSize==0: 
            TestData = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/'+TestingFileInput+'.csv',header = None)
            X_test, y_test = TestData[:,:-1],TestData[:,-1]
            X_train, y_train = X, y
            print("N_training examples : ",X_train.shape[0])
            print("N_test examples : ",X_test.shape[0])
        elif TestingFileInput == '':
            X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=TestSize)
            print("N_training examples : ",X_train.shape[0])
            print("N_test examples : ",X_test.shape[0])
        MLModel = RandomForestClassifier(n_estimators=100, random_state=0)
        MLModel.fit(X_train,y_train)
        print(MLModel.score(X_train,y_train))
        print(MLModel.score(X_test,y_test))