import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unittest
from PIL import Image
#from sklearn.externals import joblib
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


NUMERIC_FEATURE_SPLIT = 1
NAME_CARDS_INDEX = 0
PREDICTION_INDEX = 0
HEADER_PRESENT = 0
NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN = 7
LIST_OF_MARKERS_PLT = ['-','--','-.',':']

class Utility:

    def __init__(self) -> None:

        """

        Utility class is a tool class to : 

        * read the data files using pandas
        * display cards images using matplotlib : TODO update this function
        * Lower the resolution of the images in case the display is taking too much time.

        

        """

        self.DPI_DISPLAY_PREDICTION = 40
        self.DPI_SHOW_HAND = 50
        self.FIG_SIZE = (75,75)


    def read(self,
             path: str,
             header: int = HEADER_PRESENT
             ) -> np.array:

        """
        TODO : the header is NOT ALWAYS an int and should be always the same variable type
        Function reading the excel data files (TODO precise this in the function name)

        Parameters
        ----------
        path
            path to the excel file to be read 

        header = HEADER_PRESENT means that there is a header, put header = None if there is no header
        """
        return np.copy(pd.read_csv(path,sep=';',header=header))
    
    def _DisplayImage(self,
                      Image: np.hstack,
                      FigSize: (int,int),
                      Dpi: int
                      ) -> None:
        """
        TODO : WRITE THE IMAGE TYPE the Image is a h.stack type, no Idea exactly what it is and it should be more clear
        Function displaying images (stack Images or jpg ? uses which module to read the images ? plt)

        Parameters
        ----------
        Image
            Image to be displayed
        FigSize 
            Size of the figure/image, TODO check what it is exactly
        Dpi 
            dpi of the image to be displayed

        """                      
        plt.figure(figsize = FigSize, dpi = Dpi)
        plt.imshow(Image)
        plt.axis('off')
        plt.show()

    def LowerResolution(self) -> None:
        """
        Functions that reduces the constant variables used for displaying images in case the display is too long
        """

        self.DPI_DISPLAY_PREDICTION = 4
        self.DPI_SHOW_HAND = 4
        self.FIG_SIZE = (200,200)

class Deck:

    def __init__(self,
                 DeckName: str,
                 DeckDict: dict = {str : int},
                 ) -> None:

        """
        Initialisation of the class Deck 

        Parameters
        ----------
        DeckName
            The name of the deck
        DeckDict
            The dictionary corresponding to the cards inside the deck and their number.
            The keys of the dictionnary are the cards written as str, and the corresponding values are their number.
        """

        
        self.DeckDict = DeckDict
        self.DeckName = DeckName

        # Looks for the features excel sheet of each card.
        self.PathToFeatures: str = self.DeckName+'/Training_set_'+self.DeckName+'/Features.csv'
        self.Features: np.array = Utility.read(self,self.PathToFeatures)[:,NUMERIC_FEATURE_SPLIT:]
        self._FeaturesShape = self.Features.shape

        # calls the function _decklist
        self.DeckList = Deck._decklist(self)

        # calls the function _cardlist
        self.CardList = Deck._cardlist(self) 

    def _decklist(self) -> list:

        """
        _decklist function returns a list of all the cards (str) of the deck containing all the cards including duplicates cards.

        """

        DeckList: list = []
        Cards = self.DeckDict.items()
        for Card, NumberOfCopy in Cards:
            for _ in range(NumberOfCopy):
                DeckList.append(Card)
        return DeckList

    def _cardlist(self) -> list:

        """
        _cardlist function returns a list of all the cards (str) containing all the cards of the feature excel sheet without duplicates.

        """
        CardList = Utility.read(self,self.PathToFeatures)[:,NAME_CARDS_INDEX]
        CardList = list(CardList)
        return CardList

class ML:

    def __init__(self,
                 DeckName: str,
                 ) -> None:
        """
        Initialisation of the class ML (Machine Learning) 

        Parameters
        ----------
        DeckName
            The name of the deck, it must be the same than for the class deck in order to load the correct model
        """                                                   
        self.DeckName: str = DeckName
    
    def LoadModel(self)-> None:
        self.PathToSavedWeights: str = self.DeckName+'/Training_set_'+self.DeckName+'/'+self.DeckName+'SavedWeights.pkl'
        self.ModelML = joblib.load(self.PathToSavedWeights)

class Main:

    def __init__(self,
                 DeckName: str,
                 DeckDict: dict = {str : int},
                 Resolution: str ='high'
                 ) -> None:
        """
        Initialisation of the class Main 

        Parameters
        ----------
        DeckName
            The name of the deck, it must be the same than for the class deck in order to load the correct model
        DeckDict
            The dictionary corresponding to the cards inside the deck and their number.
            The keys of the dictionnary are the cards written as str, and the corresponding values are their number.
        Resolution
            the resolution of the cards displayed, "high" by default and can be changed to "low"
        """    
        # Creates an instance of Deck class
        CurrentDeck = Deck(DeckName,DeckDict)

        # Creates an instance of ML class
        CurrentDeckMLModel = ML(DeckName)
        CurrentDeckMLModel.LoadModel()

        # Creates an instance of the Utility class
        CurrentResolution = Utility()
        if Resolution not in ("low","high"):
            print("Resolution has to be the str 'low' or the str 'high', here it's : {}".format(Resolution))
            return
        if Resolution == 'low':
            CurrentResolution.LowerResolution()
        
        # Stores all the Current deck attributes
        self.DeckDict = CurrentDeck.DeckDict
        self.DeckList = CurrentDeck.DeckList
        self.CardList = CurrentDeck.CardList
        self.Features = CurrentDeck.Features
        self.DeckName = CurrentDeck.DeckName

        # Stores the ML current deck model attribute
        self.ModelML = CurrentDeckMLModel.ModelML

        # Stores the Utility attributes
        self.DPI_SHOW_HAND = CurrentResolution.DPI_SHOW_HAND
        self.DPI_DISPLAY_PREDICTION = CurrentResolution.DPI_DISPLAY_PREDICTION
        self.FIG_SIZE = CurrentResolution.FIG_SIZE


    def _CardsToIndex(self,
                      DeckList: list,
                      ) -> list:
        """
        _CardsToIndex function returns a list of all the cards numbers (int) of the deck containing all the cards including duplicates cards.
        It creates the list of integers DeckListNumbers corresponding to DeckList.

        The goal of this function is later on to be able to shuffle the list using the np.random funtion which only works on numbers.

        Parameters
        ----------
        DeckList
            list of all the cards (str) of the deck containing all the cards including duplicates cards.
        """
        DeckListNumbers: list = []
        for Card in DeckList:
            DeckListNumbers.append(self.CardList.index(Card))
        return DeckListNumbers

    def _CreateHand(self,
                    DeckList: list,
                    n: int
                    ) -> list:
        """
        Creates a hand from the DeckList of cards with n cards in it, which is a list of n strings containing the names of the card.
        This structure will be called HandNames later in the code.

        Parameters
        ----------
        DeckList
            list of all the cards (str) of the deck containing all the cards including duplicates cards.

        """

        if n > len(DeckList):
            # TODO use the errors handling already inplemented in python
            print("Error, n > len(DeckList), ({0} > {2}) the function CreateHand cannot create a hand with {0} cards from a DeckList"
            " only composed by the following cards {1}".format(n, DeckList, len(DeckList)))
            return          

        # To shuffle the list, the list must be composed of numbers, hence the _CardsToIndex function
        DeckListNumbers = Main._CardsToIndex(self,DeckList) 

        # Here the list is shuffled
        DeckListNumbersShuffled = [i for i in DeckListNumbers]
        np.random.shuffle(DeckListNumbersShuffled)

        # The hand is drawn from the beginning of the list 
        HandNumbers = DeckListNumbersShuffled[0:n]
        HandNames = [self.CardList[i] for i in HandNumbers]

        return HandNames 

    def CreateHandFromDicts(self,
                            DictList: list = [],
                            nList: list = [NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN]
                            ) -> list:
        """
        This function creates a hand composed by nlist[i] cards from the dictionary DictList[i] 
        by creating several instances of the class Deck for each i

        Parameters
        ----------
        
        DictList
            List of python dictionnaries you want to create your hand with.
        nList
            List of number of cards (integers) you want to pick from each dictionnary in DictList, respectively.

        """

        if DictList == []:
            DictList = [self.DeckDict]

        len_DictList = len(DictList)
        len_nList = len(nList)

        if len(self.DeckList) == 0:
            print('Error, DeckList is empty, perhaps DeckDict has been forgotten in Main ?')
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
        if np.sum(nList) != NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN:
            print('Error, The sum of nList is not equal to the numbers of cards without mulligan, which is 7 !')
            return

        ListOfDeckLists: list = []
        for i in range(len_nList):
            SubDeck_i = Deck(self.DeckName,DictList[i])
            ListOfDeckLists.append(SubDeck_i.DeckList)

        HandNames: list = []
        for i in range(len_nList):
            HandNames += Main._CreateHand(self,ListOfDeckLists[i],nList[i])
        return HandNames


    def SortHand(self,
                 HandNames: list
                 ) -> list:
        """
        This function sorts a hand that is a list of str (the different cards in it),
        according to the list defined in the features.csv file.
        It return an output which is the sorted list of str.

        Parameter
        ----------
        
        HandNames
            list of str (the different cards in it)
        """

        HandToSortNumbers = [self.CardList.index(HandNames[i]) for i in range(len(HandNames))]
        HandToSortNumbers.sort()
        SortedHandNames = [self.CardList[i] for i in HandToSortNumbers]
        return (SortedHandNames)

    def _ConvertCardIntoFeatures(self,
                                 Card: str
                                 ) -> np.array:
        """
        This function takes a card that is a str as an input and 
        returns a 1D-numpy array corresponding to the features of the card
        according features.csv file.

        Parameter
        ----------
        
        Card
            the str corresponding to the card.
        """

        CardIndex = self.CardList.index(Card)
        CardFeature = self.Features[CardIndex]
        return(CardFeature)
    
    def _MakeTestableHand(self,
                          HandNames: list
                          ) -> np.array:

        """
        This function takes a hand that is a list of str (the different cards in it) as an input and 
        returns a 2D-numpy array corresponding to the features of the cards concatenated 
        according features.csv file.
        Thus this function makes the output ready for the Scikit-Learn.predict() function.

        Parameter
        ----------
        
        HandNames
            the list of str corresponding to the cards in the hand.
        """
        TestableHand = []
        for i in range(len(HandNames)):
            Card_i = HandNames[i]
            CardFeature_i = Main._ConvertCardIntoFeatures(self,Card_i)
            TestableHand = np.concatenate((TestableHand,CardFeature_i))
        TestableHand = [TestableHand] # in order to have a 2D array to run the algorithm
        return(TestableHand)

    def ShowHand(self,
                 HandNames: list
                 ) -> None:

        """
        This function takes a list of str as input and 
        displays the images corresponding to the sorted cards in hand.
        TODO Clean the function

        Parameter
        ----------
        
        HandNames
            the list of str corresponding to the cards in the hand.        
        """
        ImagesList = []
        for Card in HandNames:
            ImagesList.append(self.DeckName+'/Pictures_'+self.DeckName+'/'+Card+'.jpg')
        Images = [ Image.open(i) for i in ImagesList ]
        MinimumShape = sorted( [(np.sum(i.size), i.size ) for i in Images])[0][1]
        ImagesCombined = np.hstack( (np.asarray( i.resize(MinimumShape) ) for i in Images ) )
        Utility._DisplayImage(self,ImagesCombined,self.FIG_SIZE,self.DPI_SHOW_HAND)
        

    def _displayPrediction(self,
                           prediction: int
                           ) -> None :
        """
        This function takes a prediction (int) as an input and 
        displays the corresponding image (Keep or Mulligan)

        Parameter
        ----------
        
        prediction
            the int corresponding to the result of the algorithm       
        """

        if prediction == 1:
            Img =  Image.open('General/Pictures/Keep.PNG')
            Utility._DisplayImage(self,Img,self.FIG_SIZE,self.DPI_DISPLAY_PREDICTION)

        if prediction == 0:
            Img =  Image.open('General/Pictures/Mulligan.PNG')
            Utility._DisplayImage(self,Img,self.FIG_SIZE,self.DPI_DISPLAY_PREDICTION)

    def TestHand(self,
                 HandNames: list
                 ) -> int: 
        """
        This function takes a HandNames (list of str) as an input and 
        returns the corresponding prediction using the algorithm (0 or 1)

        Parameter
        ----------
        
        HandNames
            the list of str corresponding to the cards in the hand.   
        """

        TestableHand = Main._MakeTestableHand(self,HandNames)
        prediction = self.ModelML.predict(TestableHand)[PREDICTION_INDEX]
        Main._displayPrediction(self,prediction)
        return prediction

    def RunHand(self,
                Hand: str
                ) -> None:

        """
        This function takes a Hand (str with spaces between the card names) as an input and 
        displays the hand with corresponding image (Keep or Mulligan)

        Parameter
        ----------
        
        HandNames
            str of the concatenation of all cards contained in the Hand with spaces between names
        """
        HandNames = Hand.split()
        SortedHandNames = Main.SortHand(self,HandNames)
        Main.ShowHand(self,SortedHandNames)
        Main.TestHand(self,SortedHandNames)

    def TestModel(self,
                  n: int,
                  DictList: list = [],
                  nList: list = [NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN]
                  ) -> None:
        """
        This function tests the current algorithm on n hands generated using nList[i] (int) cards from the DictList[i] (dict)

        Parameters
        ----------
        
        n
            number of hands to be tested
        DictList
            list of dictionnaries that will generate the hands
        nList
            list of integers that will represent the number of cards drawn from each dictionnary    
        """
        for _ in range(n):  
            HandNames = Main.CreateHandFromDicts(self,DictList=DictList,nList=nList)
            Hand = ' '.join(HandNames)
            Main.RunHand(self,Hand)

class Train:

    def __init__(self,
                 DeckName: str,
                 DeckDict: dict = {str : int},
                 Resolution: str ='high'
                 ) -> None:

        """
        Initialisation of the class Train 

        Parameters
        ----------
        DeckName
            The name of the deck
        DeckDict
            The dictionary corresponding to the cards inside the deck and their number.
            The keys of the dictionnary are the cards written as str, and the corresponding values are their number.
        Resolution
            str that represent the quality of the images printed by the function. The str must be equal to 'low' or to 'high'. 
        """

        CurrentDeck = Deck(DeckName,DeckDict)
        CurrentResolution = Utility()
        if Resolution not in ("low","high"):
            print("Resolution has to be the str 'low' or the str 'high', here it's : {}".format(Resolution))
            return
        if Resolution == 'low':
            CurrentResolution.LowerResolution()
        self.DeckName = CurrentDeck.DeckName
        self.DeckList = CurrentDeck.DeckList
        self.CardList = CurrentDeck.CardList
        self.Features = CurrentDeck.Features
        self.DPI_SHOW_HAND = CurrentResolution.DPI_SHOW_HAND
        self.DPI_DISPLAY_PREDICTION = CurrentResolution.DPI_DISPLAY_PREDICTION
        self.FIG_SIZE = CurrentResolution.FIG_SIZE

    def MyScore(self,
                TestingFileInput: str ='TestingSet'
                ) -> list:
        """
        This function tests the current algorithm on the test set testingFileInput 
        and returns the failed hands.

        Parameters
        ----------
        TestingFileInput
            The str that designs the test data file
        """

        CurrentDeckMLModel = ML(self.DeckName)
        CurrentDeckMLModel.LoadModel()
        self.ModelML = CurrentDeckMLModel.ModelML
        FailedHands = []
        TestDataNames = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/'+TestingFileInput+'Names.csv',header=None)
        TestData = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/'+TestingFileInput+'.csv',header=None)
        X_test_names = TestDataNames[:,:-1]
        X_test, y_test = TestData[:,:-1],TestData[:,-1]
        N_test_examples = X_test.shape[0]
        print("N_test examples : ",N_test_examples)
        for i in range(N_test_examples) :
            TestableHand = X_test[i].reshape(1, -1)
            prediction = self.ModelML.predict(TestableHand)[PREDICTION_INDEX]
            if prediction != y_test[i]:
                FailedHands.append(" ".join(X_test_names[i]))
        PercentageScore = 100*(1-(len(FailedHands)/N_test_examples))
        print("Score is : {} % \n".format(np.around(PercentageScore,decimals = 2)))
        return FailedHands
                

    def MakeTrainingSet(self,
                        n: int,
                        TrainingSetSize: int,
                        TrainingFileName: str
                        ) -> None:
        """
        This function helps the user creating the data sets by creating 
        TrainingSetSize data points (hands) and then writes them to the file TrainingFileName

        Inputing 9 breaks the loop and finishes the function

        Parameters
        ----------
        n   
            number of cards in the hands generated.
        TrainingSetSize
            number of hands generated.
        TrainingFileName
            The str that designs the training data file that will be used to write the hands
        """

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

    def MakeTrainingSetWithModel(self,
                                 n: int,
                                 TrainingSetSize: int,
                                 TrainingFileName: str
                                 ) -> None:
        """
        This function helps the user creating the data sets by creating 
        TrainingSetSize data points (hands) and then writes them to the file TrainingFileName.
        Contrarely with the MakeTrainingSet function, this function uses the algorithm itself to 
        assess the hands and asks the user if the algorithm correctly classified the hands. 

        Inputing 9 breaks the loop and finishes the function

        Parameters
        ----------
        n   
            number of cards in the hands generated.
        TrainingSetSize
            number of hands generated.
        TrainingFileName
            The str that designs the training data file that will be used to write the hands
        """                                 
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


    def MakeControlledTrainingSetWithModel(self,
                                           n: int,
                                           DictList: list,
                                           nList: list,
                                           TrainingSetSize: int,
                                           TrainingFileName: str
                                           ) -> None:

        """
        This function helps the user creating the data sets by creating 
        TrainingSetSize data points (hands) and then writes them to the file TrainingFileName.
        This function uses the algorithm itself to assess the hands and asks the user
        if the algorithm correctly classified the hands. Moreover this function generates the hands 
        according to the list of dictionnaries DictList and the list of ints nList

        Inputing 9 breaks the loop and finishes the function

        Parameters
        ----------
        n   
            number of cards in the hands generated.
        DictList 
            represents the list of python dictionnaries you want to create your hand with.
        nList 
            represents the list of number of cards (integers) you want to pick from each dictionnary in DictList, respectively.
        TrainingSetSize
            number of hands generated.
        TrainingFileName
            The str that designs the training data file that will be used to write the hands
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

    def _TrainingSetToDocs(self,
                          TrainingFileInput: str
                          ) -> (list,list):
        """
        This function generates the Docs and Labels to train the sklearn algorithm

        Parameters
        ----------

        TrainingFileInput
            The str that designs the training data file that will be read to generate the docs
        """        


        TrainingSet = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/'+TrainingFileInput+'.csv',header=None)
        Docs = []
        for Line in TrainingSet:
            Line = Line[0:NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN]
            Docs.append(" ".join(Line))
        Labels = TrainingSet[:,NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN]   
        return (Docs,Labels)


    def _WriteTrainingSetFeatureFromDocs(self,
                                        Docs: list,
                                        Labels: list,
                                        TrainingFileOutput: str
                                        ) -> None:
        """
        This function writes the TrainingFileOutput.csv file using the features of the cards concatenated into the hands
        (Docs) and the labels of those hands (Labels)

        Parameters
        ----------
        Docs
            list of all the hands (as features with numbers) needed to be written in the file
        Labels
            list of all the labels corresponding to the hands needed to be written in the file
        TrainingFileOutput
            The str that designs the training data file that will be written
        """                                         
        pv =';'
        with open(self.DeckName+'/Training_set_'+self.DeckName+'/'+TrainingFileOutput+'.csv' , 'w') as TrainingFile:
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

    def TransformTrainingSet(self,
                             TrainingFileInput: str,
                             TrainingFileOutput: str= 'TrainingSet'
                             ) -> None:
        """
        This function writes the TrainingFileOutput.csv file (with numbers instead of str) 
        using the features of the cards concatenated into the hands contained the 
        TrainingFileInput file

        Parameters
        ----------
        TrainingFileInput
            The str that designs the training data file that will be read to generate the docs
        TrainingFileOutput
            The str that designs the training data file that will be written
        """         

        Docs,Labels = Train._TrainingSetToDocs(self,TrainingFileInput)
        Train._WriteTrainingSetFeatureFromDocs(self,Docs,Labels,TrainingFileOutput)

    def TrainAndSaveWeights(self,
                            Nestimators: int = 100,
                            MaxDepth: int = None,
                            save: bool = True
                            ) -> None:
                            
        """
        This function trains the random forest classifier and saves the weights of the classifier

        Parameters
        ----------
        Nestimators
            integer that designs the number of trees used for the algorithm
        MaxDepth
            The maximum depth of each tree. 
        save
            boolean variable to save or not the weights.
        """    

        Data = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/TrainingSet.csv',header = None)
        X, y = Data[:,:-1],Data[:,-1]
        N_examples = X.shape[0]
        N_features = X.shape[1]
        print("N_examples : ",N_examples)
        print("N_features : ",N_features)
        MLModel = RandomForestClassifier(n_estimators=Nestimators,max_depth=MaxDepth, random_state=0)
        MLModel.fit(X, y)
        print(MLModel.score(X,y))
        self.ModelML = MLModel
        if save:
            joblib.dump(MLModel,self.DeckName+'/Training_set_'+self.DeckName+'/'+self.DeckName+'SavedWeights.pkl')
    
    def FindBestNestimator(self,
                           X_train: np.array,
                           y_train: np.array,
                           X_test: np.array,
                           y_test: np.array,
                           MaxDepth: int,
                           N0: int = 30,
                           NestimatorsList: list = []
                           ) -> (int,list,list,list):
        """
        This function finds the best value for Nestimator according to the test score and returns the best value for Nestimator.
        Moreover it returns the lists of the values of Nestimator used for the calibration, as well as the corresponding lists of 
        Score (testing score) and fit (training score). The values used for Nestimator are incremented by 5 in the loop.

        Parameters
        ----------
        X_train
            X data used for training, 2D np.array
        y_train
            y data (labels) used for training, 2D np.array
        X_test
            X data used for testing, 2D np.array
        y_test
            y data (labels) used for testing, 2D np.array
        MaxDepth
            The maximum depth of each tree.
        N0
            minimum value for Nestimator used if the NestimatorsList is not inputed.
        NestimatorsList
            list of the Nestimators values to be tested
        """                              
        increment = 5
        InitialNestimator = N0
        FinalNestimator = 300
        ListNestimator,ListScore,ListFit = [], [], []
        print("Finding the best N_estimators ...")
        if NestimatorsList == []:
            for i in range((FinalNestimator-InitialNestimator)//increment):
                Nestimators = InitialNestimator + increment*i
                MLModel = RandomForestClassifier(n_estimators = Nestimators, max_depth = MaxDepth,  random_state=0)
                MLModel.fit(X_train,y_train)
                Score = MLModel.score(X_test,y_test)
                fit = MLModel.score(X_train,y_train)
                ListFit.append(fit)
                ListNestimator.append(Nestimators)
                ListScore.append(Score)
        else :
            for Nestimators in NestimatorsList:
                MLModel = RandomForestClassifier(n_estimators = Nestimators, max_depth = MaxDepth,  random_state=0)
                MLModel.fit(X_train,y_train)
                Score = MLModel.score(X_test,y_test)
                fit = MLModel.score(X_train,y_train)
                ListFit.append(fit)
                ListNestimator.append(Nestimators)
                ListScore.append(Score)
        BestScore = np.max(ListScore) 
        BestScoreIndex = ListScore.index(BestScore)
        BestNestimator = ListNestimator[BestScoreIndex]
        print("Best Score found : {} , Best N_estimators found : {} \n".format(np.around(BestScore,decimals=4) ,BestNestimator))
        return BestNestimator, ListNestimator, ListScore , ListFit


    def TrainAndTest(self,
                     Nestimators: int = 100,
                     FindBestNestimators: bool = True,
                     NestimatorsList: list = [],
                     TestSize: float = 0,
                     MaxDepth: int = None,
                     TestingFileInput: str = ''
                     ) -> (list,list,list):
        """
        This function trains the algorithm while searching for the best value of Nestimator if FindBestNestimators is true.
        Moreover it returns the lists of the values of Nestimator used for the calibration, as well as the corresponding lists of 
        Score (testing score) and fit (training score).

        Parameters
        ----------
        Nestimators
            number of trees used for the random forest classifier
        FindBestNestimators
            finds the best value for Nestimator if True
        NestimatorsList
            list of the Nestimators values to be tested
        TestSize
            the separation between train and test data (0.2 corresponds to 20% of test data and 80% of training data) 
        MaxDepth
            The maximum depth of each tree.
        TestingFileInput
            The testing file if selected
        """

        Data = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/TrainingSet.csv',header = None)
        X, y = Data[:,:-1],Data[:,-1]
        print("N_examples : ",X.shape[0])
        if TestSize==0: 
            TestData = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/'+TestingFileInput+'.csv',header = None)
            X_test, y_test = TestData[:,:-1],TestData[:,-1]
            X_train, y_train = X, y
            print("N_training examples : {}".format(X_train.shape[0]))
            print("N_test examples : {} \n".format(X_test.shape[0]))
        elif TestingFileInput == '':
            X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=TestSize)
            print("N_training examples : {} ".format(X_train.shape[0]))
            print("N_test examples : {} \n".format(X_test.shape[0]))
        if FindBestNestimators:
            Nestimators, ListNestimator, ListScore, ListFit \
            = Train.FindBestNestimator(self,X_train,y_train,X_test,y_test,MaxDepth=MaxDepth , NestimatorsList = NestimatorsList)
        MLModel = RandomForestClassifier(n_estimators=Nestimators, max_depth = MaxDepth, random_state=0)
        MLModel.fit(X_train,y_train)
        print("Training Score : {} ".format(np.around(MLModel.score(X_train,y_train),decimals = 4)))
        print("Testing Score : {} \n".format(np.around(MLModel.score(X_test,y_test),decimals = 4)))
        if FindBestNestimators: 
            return ListNestimator, ListScore, ListFit
        

class Analyse:

    def __init__(self,
                 DeckName: str,
                 DeckDict: dict = {str : int}
                 ) -> None:
        """
        Initialisation of the class Analyse 

        Parameters
        ----------
        DeckName
            The name of the deck, it must be the same than for the class deck in order to load the correct model
        DeckDict
            The dictionary corresponding to the cards inside the deck and their number.
            The keys of the dictionnary are the cards written as str, and the corresponding values are their number.
        """    

        CurrentDeck = Deck(DeckName,DeckDict)
        CurrentDeckMLModel = ML(DeckName)
        CurrentDeckMLModel.LoadModel()

        self.DeckDict = CurrentDeck.DeckDict
        self.DeckList = CurrentDeck.DeckList
        self.CardList = CurrentDeck.CardList
        self.Features = CurrentDeck.Features
        self._FeatureShape = CurrentDeck._FeaturesShape
        self.DeckName = CurrentDeck.DeckName
        self.ModelML = CurrentDeckMLModel.ModelML

    def AnalysePattern(self
                       ) -> None :

        """
        This function analyses the results of the algorithm and the 10 features that are the most important.
        """
                       
        self.ImportanceFeature = list(self.ModelML.feature_importances_)
        ImportanceFeatureSorted = self.ImportanceFeature
        ImportanceFeatureSorted.sort(reverse=True)

        NumberOfFeaturesPerCardIndex = 1
        FeaturesNameRaw = 1
        FeaturesNameLignIndex = 0

        NumberOfFeaturesPerCard = self.Features.shape[NumberOfFeaturesPerCardIndex]
        FeatureNames = Utility.read(self,self.DeckName+'/Training_set_'+self.DeckName+'/Features.csv',header = None)[FeaturesNameLignIndex]
        FeatureNames = FeatureNames[FeaturesNameRaw:]

        if len(self.ImportanceFeature)!=NumberOfFeaturesPerCard*NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN:
            ErrorSentence = """Error in the Number of Features ! The model has {} features per hand and the loaded
                                features.csv has {}*{} = {} features per hand"""
            print(ErrorSentence.format(len(self.ImportanceFeature),NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN,
            NumberOfFeaturesPerCard,NumberOfFeaturesPerCard*NUMBERS_OF_CARDS_IN_HAND_NO_MULLIGAN))
            return

        count = 1
        for ImportanceOfFeature in ImportanceFeatureSorted:
            FeatureIndex = self.ImportanceFeature.index(ImportanceOfFeature)
            NumberOfFeature = FeatureIndex % NumberOfFeaturesPerCard 
            NumberOfCard =  FeatureIndex // NumberOfFeaturesPerCard + 1
            Feature = FeatureNames[NumberOfFeature]
            ResultSentence = """The N°{} feature is the feature : "{}" of the card N°{}  with the score of {} percents """
            print(ResultSentence.format(count,Feature,NumberOfCard,np.around(100*ImportanceOfFeature,decimals=2)))
            count += 1
            if count > 10 :
                break

    def PlotGraphs(self,
                   MaxDepthList: list,
                   TestSizeList: list,
                   NestimatorsList: list,
                   Nexperiments: int = 50
                   ) -> None:

        """
        This function plots the graphs of the test score against the max depth (fixed Nestimator) and against the value Nestimator
        (fixed max depth)

        Parameters
        ----------
        MaxDepthList
            list of the different Max Depth values to be tested
        TestSizeList
            list of the different values of the testSize separation to be tested
        NestimatorsList
            list of the Nestimators values to be tested
        Nexperiments
            number of experiements
        """

        Results_testing = np.zeros((Nexperiments,len(MaxDepthList),len(NestimatorsList)))
        Results_training = np.zeros((Nexperiments,len(MaxDepthList),len(NestimatorsList)))
        for _,TestSize in enumerate(TestSizeList):
            for MaxDepth_idx, MaxDepth in enumerate(MaxDepthList):
                for count in range(Nexperiments):
                    _, ListScore, ListFit \
                        = Train.TrainAndTest(self,FindBestNestimators=True,NestimatorsList = NestimatorsList,MaxDepth = MaxDepth,TestSize=TestSize)
                    
                    Results_testing[count,MaxDepth_idx,:] = ListScore
                    Results_training[count,MaxDepth_idx,:] = ListFit

                    print('{} / {}'.format(count+1 + MaxDepth_idx*Nexperiments ,Nexperiments*len(MaxDepthList)))


        Results_testing_avg = np.mean(Results_testing,axis = 0)
        Results_training_avg = np.mean(Results_training,axis = 0)

        Legend = []
        for count in range(len(MaxDepthList)):
            Marker = LIST_OF_MARKERS_PLT[count % len(LIST_OF_MARKERS_PLT)]

            plt.plot(NestimatorsList,Results_testing_avg[count,:],'r'+ Marker ,
                NestimatorsList,Results_training_avg[count,:],'b'+ Marker)
            Legend.append('Testing score / MaxDepth ='+str(MaxDepthList[count]))
            Legend.append('Training score / MaxDepth ='+str(MaxDepthList[count]))
        plt.legend(Legend,shadow=True, loc=(1, 0.7), handlelength=1.5, fontsize=16)
        plt.xlabel('N_estimators')
        plt.ylabel('Score in %')
        plt.show()

        Legend = []
        for count in range(len(NestimatorsList)):
            Marker = LIST_OF_MARKERS_PLT[count % len(LIST_OF_MARKERS_PLT)]

            plt.plot(MaxDepthList,Results_testing_avg[:,count],'r'+ Marker ,
                MaxDepthList,Results_training_avg[:,count],'b'+ Marker)
            Legend.append('Testing score / N_estimators ='+str(NestimatorsList[count]))
            Legend.append('Training score / N_estimators ='+str(NestimatorsList[count]))
        plt.legend(Legend,shadow=True, loc=(1, 0.7), handlelength=1.5, fontsize=16)
        plt.xlabel('Max Depth')
        plt.ylabel('Score in %')
        plt.show() 