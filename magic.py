import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

### group 1 :  deck dict, 
class Deck:

    def __init__(self,DeckDict,DeckName):
        self.DeckDict = DeckDict
        self.DeckName = DeckName
        self.Features = np.copy(pd.read_csv('Training_set_'+self.DeckName+'/Features.csv',sep=';'))[:,1:]
        self.DeckList = Deck.decklist(self)
        self.CardList = Deck.cardlist(self) 

    def decklist(self):
        deck_list =[]
        cards = self.DeckDict.items()
        for card_itself,number_of_the_card in cards:
            for _ in range(number_of_the_card):
                deck_list.append(str(card_itself))
        return deck_list

    def cardlist(self):
        card_list = list(self.DeckDict.keys())
        return card_list

    def GetDeckName(self):
        return self.DeckName

    def GetDeckList(self):
        return self.DeckList

    def GetCardList(self):
        return self.CardList

    def GetFeatures(self):
        return self.Features


class ML:

    def __init__(self,DeckName):
        self.DeckName = DeckName
        self.ModelML = ML.LoadModel(self)
    
    def LoadModel(self):
        return joblib.load('Training_set_'+self.DeckName+'/'+self.DeckName+'SavedWeights.pkl')


### group 2 : create hand & turn it into testable hand """class hand ?"""

class Main:

    def __init__(self,DeckDict,DeckName):
        CurrentDeck = Deck(DeckDict,DeckName)
        self.DeckList = CurrentDeck.GetDeckList()
        self.CardList = CurrentDeck.GetCardList()
        self.Features = CurrentDeck.GetFeatures()
        self.DeckName = CurrentDeck.GetDeckName()
        self.ModelML = ML(DeckName).ModelML


    def CardsToIndex(self):
        card_index = []
        deck_list_set = list(set(self.DeckList))
        for card in self.DeckList:
            card_index.append(deck_list_set.index(card))
        return card_index,deck_list_set

    def CreateHand(self,n):
        deck_numbers_index,deck_list_map = Main.CardsToIndex(self.DeckList) 
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
            list_im.append('images_'+self.DeckName+'/'+word.lower()+'.jpg')
        imgs    = [ Image.open(i) for i in list_im ]
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
        imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
        plt.figure(figsize=(20, 16), dpi= 100, facecolor='w', edgecolor='k')
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


    def TestHand(self,testable_hand): 
        testable_hand_RFC = Main.TestableHandForRFC(self,testable_hand)
        if self.ModelML.predict(testable_hand_RFC)[0] ==1:
            print('-------------------------------------------------------KEEP-------------------------------------------------------','\n')
            print('\n')
            print('\n')
        else :
            print('-------------------------------------------------------MULL-------------------------------------------------------','\n')
            print('\n')
            print('\n')

    def RunHand(self,hand):
        sorted_hand = Main.SortHand(self,hand[0].split())
        testable_hand = Main.CreatedHandToTestableHand(self,sorted_hand)
        Main.ShowHand(self,testable_hand)
        Main.TestHand(self,testable_hand)

    def Test_model(self,deck_dict,n_test):
        for _ in range(n_test):  
            created_hand = Main.CreateHand(self,7)
            testable_hand = Main.CreatedHandToTestableHand(self,created_hand)
            Main.RunHand(self,testable_hand)

class Training:

    def __init__(self,DeckDict,DeckName,FeaturesRaw):
        CurrentDeck = Deck(DeckDict,DeckName)
        self.DeckList = CurrentDeck.GetDeckList()
        self.CardList = CurrentDeck.GetCardList()
        self.Features = CurrentDeck.GetFeatures()

    def training_set(self,n,TrainingSetSize,TrainingFileName):
        pv =';'
        with open(TrainingFileName+'.csv' , 'w') as TrainingFile,\
        open(TrainingFileName+'question.csv' , 'w') as QuestionFile:
            for _ in range(TrainingSetSize):
                current_hand = Main.CreateHand(self,n)
                sorted_hand = Main.SortHand(self,current_hand)
                testable_sorted_hand = Main.CreatedHandToTestableHand(self,sorted_hand)         
                Main.ShowHand(self,testable_sorted_hand)           
                y = input("Keep: 1, Mull: 0 or Not sure: 3 ?")
                print('\n')
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

    def training_set_to_docs_and_labels(self,TrainingFileName):
        training_set = pd.read_csv(TrainingFileName+'.csv',sep =';',header=None)
        training_set = np.copy(training_set)
        docs = []
        for line in training_set:
            line = line[0:7]
            docs.append(" ".join(line))
        labels = training_set[:,7]           
        return(docs,labels)

    def new_features_from_docs(self,docs,labels,training_file_name):
        pv =';'
        with open(training_file_name+'.csv' , 'w') as training_file:
            for i in range(len(docs)):
                docs_i = Main.SortHand(self,docs[i].split())
                line_to_write=''
                for j in range(7):
                    feature_card = Main.ConvertCardIntoFeatures(self,docs_i[j])
                    K = len(feature_card)
                    for k in range(K):
                        line_to_write += str(int(feature_card[k]))+pv
                line_to_write+=str(labels[i])
                training_file.write(line_to_write+'\n')
        print('Done !')  