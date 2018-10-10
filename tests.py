import numpy as np
from PIL import Image
import unittest
import magic

class UtilityTest(unittest.TestCase):

    def setUp(self):
        self.DeckName = 'Test'
    
    def test_Read(self):
        PATH = self.DeckName+'/Training_set_'+self.DeckName+'/Features.csv'
        READ_CSV =  [['sacred_foundry',0,1,2,3,4],
                    ['wooded_foothills',59,5,6,7,8],
                    ['mountain',87,9,10,11,12],
                    ['stomping_ground',66,13,14,15,16]]
        vertical_shape = 4
        horizontal_shape = 6            
        READ_CSV_SHAPE = (vertical_shape,horizontal_shape)       

        res = magic.Utility.read(self,PATH)
        self.assertEqual(READ_CSV_SHAPE,res.shape)

        for index_i in range(vertical_shape):
            for index_j in range(horizontal_shape):
                self.assertEqual(res[index_i,index_j],READ_CSV[index_i][index_j])

                if not res[index_i,index_j]==READ_CSV[index_i][index_j]:
                    print(type(res[index_i,index_j]),type(READ_CSV[index_i][index_j]))
                    
    def test_displayImage(self):
        DPI_DISPLAY_PREDICTION = 40
        FIG_SIZE = (75,75)
        Img =  Image.open('General/Pictures/Mulligan.PNG')
        magic.Utility._DisplayImage(self,Img,FIG_SIZE,DPI_DISPLAY_PREDICTION) 

class DeckTest(unittest.TestCase):
    """
    Tests the Class Deck from the magic.py file
    """


    def setUp(self):
        DeckDictTest = {"wooded_foothills" :2,
                        "stomping_ground"  :1,
                        "sacred_foundry"  :3,
                        "mountain" :4}
        TestinstanceDeck = magic.Deck('Test',DeckDictTest)
        self.features = TestinstanceDeck.Features
        self.DeckDict = TestinstanceDeck.DeckDict
        self.DeckList = TestinstanceDeck.DeckList
        self.CardList = TestinstanceDeck.CardList

    def test_features(self):
        FEATURES = np.array([[0,1,2,3,4],
                            [59,5,6,7,8],
                            [87,9,10,11,12],
                            [66,13,14,15,16]])
        self.assertTrue((self.features==FEATURES).all())

    def test_decklist(self):
        DECKLIST = ['wooded_foothills','wooded_foothills','stomping_ground','sacred_foundry','sacred_foundry','sacred_foundry',
        'mountain','mountain','mountain','mountain']
        self.assertListEqual(self.DeckList,DECKLIST)

    def test_cardlist(self):
        CARDLIST = ['sacred_foundry', 'wooded_foothills', 'mountain', 'stomping_ground']
        self.assertListEqual(self.CardList,CARDLIST)

class MLTest(unittest.TestCase):

    def setUp(self):
        self.DeckName = 'test'

    def test_LoadModel(self):
        magic.ML.LoadModel(self)

class MainTest(unittest.TestCase):

    def setUp(self):
        DeckDictTest = {"wooded_foothills" :2,
                        "stomping_ground"  :1,
                        "sacred_foundry"  :3,
                        "mountain" :4}

        DECKLIST = ['wooded_foothills','wooded_foothills','stomping_ground','sacred_foundry','sacred_foundry','sacred_foundry',
        'mountain','mountain','mountain','mountain']

        CARDLIST = ['sacred_foundry', 'wooded_foothills', 'mountain', 'stomping_ground']

        FEATURES = np.array([[0,1,2,3,4],
                            [59,5,6,7,8],
                            [87,9,10,11,12],
                            [66,13,14,15,16]])

        TestinstanceML = magic.ML('test')
        TestinstanceML.LoadModel()
        self.Features = FEATURES
        self.DeckDict = DeckDictTest
        self.DeckName = 'test'
        self.DeckList = DECKLIST
        self.CardList = CARDLIST
        self.ModelML = TestinstanceML.ModelML
        self.NumbersOfCardsInHandNoMulligan = 7
        self.NumbersOfCardsInHandOneMulligan = 6
    
    def test_CardToIndex(self):
        DECKLISTNUMBERS = [1,1,3,0,0,0,2,2,2,2]
        self.assertListEqual(magic.Main._CardsToIndex(self,self.DeckList),DECKLISTNUMBERS)

        DECKLISTTEST = ['wooded_foothills','wooded_foothills','sacred_foundry','sacred_foundry','sacred_foundry',
        'mountain','mountain']
        DECKLISTNUMBERS = [1,1,0,0,0,2,2]
        self.assertListEqual(magic.Main._CardsToIndex(self,DECKLISTTEST),DECKLISTNUMBERS)

    def test_CreateHand(self):
        HANDNOMULLLIGAN = ['stomping_ground','mountain','sacred_foundry','mountain','wooded_foothills','mountain','mountain']
        HANDONEMULLIGAN = ['stomping_ground','mountain','sacred_foundry','mountain','wooded_foothills','mountain']

        np.random.seed(0)
        self.assertListEqual(magic.Main._CreateHand(self,self.DeckList,self.NumbersOfCardsInHandNoMulligan),
                HANDNOMULLLIGAN)

        np.random.seed(0)
        self.assertListEqual(magic.Main._CreateHand(self,self.DeckList,self.NumbersOfCardsInHandOneMulligan),
                HANDONEMULLIGAN)

    def test_CreateHandsFromDicts(self):
        HANDNOMULLLIGAN = ['stomping_ground','mountain','sacred_foundry','mountain','wooded_foothills','mountain','mountain']
        HANDCREATEDFROMDICTS = ['mountain', 'mountain', 'mountain', 'sacred_foundry', 'sacred_foundry', 'sacred_foundry', 'sacred_foundry']

        np.random.seed(0)
        self.assertListEqual(magic.Main.CreateHandFromDicts(self),HANDNOMULLLIGAN)

        DictMountain = {'mountain':1}
        DictMountain = {'mountain':1}
        DictMountain = {'mountain':1}
        DictSacredFoundry = {'sacred_foundry' : 4}
        DictList=[DictMountain,DictMountain,DictMountain,DictSacredFoundry]
        nList = [1,1,1,4]

        np.random.seed(0)
        self.assertListEqual(magic.Main.CreateHandFromDicts(self,DictList = DictList,nList = nList),HANDCREATEDFROMDICTS)


        DictMountain = {'mountain':3}
        DictSacredFoundry = {'sacred_foundry' : 4}
        DictList=[DictMountain,DictSacredFoundry]
        nList = [3,4]

        np.random.seed(0)
        self.assertListEqual(magic.Main.CreateHandFromDicts(self,DictList = DictList,nList = nList),HANDCREATEDFROMDICTS)       


    def test_SortHand(self):

        HANDTOSORT = ['mountain','sacred_foundry','wooded_foothills','stomping_ground','wooded_foothills',
        'stomping_ground','mountain']

        SORTEDHAND = ['sacred_foundry','wooded_foothills','wooded_foothills','mountain','mountain',
        'stomping_ground','stomping_ground']

        self.assertListEqual(magic.Main.SortHand(self,HANDTOSORT),SORTEDHAND)

    def test_ConvertCardIntoFeatures(self):
        CARD = 'mountain'
        FEATURESOFCARD = np.array([87,9,10,11,12])
        self.assertTrue((magic.Main._ConvertCardIntoFeatures(self,CARD)==FEATURESOFCARD).all())

        CARD = 'sacred_foundry'
        FEATURESOFCARD = np.array([0,1,2,3,4])
        self.assertTrue((magic.Main._ConvertCardIntoFeatures(self,CARD)==FEATURESOFCARD).all())

    def test_MakeTestableHand(self):
        HANDNAMES = ['mountain','sacred_foundry','wooded_foothills','stomping_ground','wooded_foothills','stomping_ground','mountain']
        TESTABLEHAND = np.array([87,9,10,11,12,0,1,2,3,4,59,5,6,7,8,66,13,14,15,16,59,5,6,7,8,66,13,14,15,16,87,9,10,11,12])
        self.assertTrue((magic.Main._MakeTestableHand(self,HANDNAMES)==TESTABLEHAND).all())

    def test_ShowHand(self):
        """
        hard to define at the moment since the 3D array representing the pictures of the cards in the hand has big shapes
        """
        #Uncomment the following lines to be sure that the Cards display correctly

        HANDNAMES = ['mountain','sacred_foundry','wooded_foothills','stomping_ground','wooded_foothills','stomping_ground','mountain']
        magic.Main.ShowHand(self,HANDNAMES) 
        pass
    
    def test_displayPrediction(self):
        """
        hard to define at the moment since That would mean testing the Imshow() function
        """
        Prediction = 0 
        magic.Main._displayPrediction(self,Prediction)

    def test_TestHand(self):
        HANDNAMES = ['mountain','sacred_foundry','wooded_foothills','stomping_ground','wooded_foothills','stomping_ground','mountain']
        self.assertIn(magic.Main.TestHand(self,HANDNAMES),[0,1])

    def test_RunHand(self):
        HAND = 'mountain sacred_foundry wooded_foothills stomping_ground wooded_foothills stomping_ground mountain'
        magic.Main.RunHand(self,HAND)


    def test_TestModel(self):
        N=1
        DictMountain = {'mountain':3}
        DictSacredFoundry = {'sacred_foundry' : 4}
        DictList=[DictMountain,DictSacredFoundry]
        n_list = [3,4]
        magic.Main.TestModel(self,N,DictList,n_list)

class TrainTest(unittest.TestCase):

    def setUp(self):
        
        DeckDictTest = {"wooded_foothills" :2,
                        "stomping_ground"  :1,
                        "sacred_foundry"  :3,
                        "mountain" :4}

        DECKLIST = ['wooded_foothills','wooded_foothills','stomping_ground','sacred_foundry','sacred_foundry','sacred_foundry',
        'mountain','mountain','mountain','mountain']

        CARDLIST = ['sacred_foundry', 'wooded_foothills', 'mountain', 'stomping_ground']

        FEATURES = np.array([[0,1,2,3,4],
                            [59,5,6,7,8],
                            [87,9,10,11,12],
                            [66,13,14,15,16]])

        TestinstanceML = magic.ML('test')
        TestinstanceML.LoadModel()
        self.Features = FEATURES
        self.DeckDict = DeckDictTest
        self.DeckName = 'test'
        self.DeckList = DECKLIST
        self.CardList = CARDLIST
        self.ModelML = TestinstanceML.ModelML
        self.NumbersOfCardsInHandNoMulligan = 7

    def test_MakeTrainingSet(self):
        pass

    def test_MakeTrainingSetWithModel(self):
        pass
    
    def test_MakeControlledTrainingSetWithModel(self):
        pass

    def test_TrainingSetToDocs(self):
        DOCS = ["sacred_foundry wooded_foothills wooded_foothills mountain mountain stomping_ground stomping_ground",
                "sacred_foundry mountain wooded_foothills mountain wooded_foothills stomping_ground stomping_ground"]
            
        LABELS = np.array([0,0])
        TrainingFileName = 'TrainingSetNames'
        Docs_index = 0
        Labels_index = 1 

        res = magic.Train.TrainingSetToDocs(self,TrainingFileName)
        self.assertListEqual(res[Docs_index],DOCS)
        self.assertTrue((res[Labels_index]==LABELS).all())

if __name__ == '__main__':
    unittest.main()