# MTGMulliganer
Here is the Magic The Gathering Mulliganer, the Machine Learning algorithm that depending on the deck you've selected, tells you if you should keep or Mulligan your hand

# Modelisation 

Each card is represented by a feature vector (that can be found in DECK_NAME/training_set_DECK_NAME/features.csv) and each hand of 7 cards is represented by the concatenation of those features vectors. The label in the training set is 0 if the hand is a Mulligan and 1 if it is a Keep.

This is the input of the Machine Learning model - which is a Random Forest Classifier - and the output is then 0 or 1 (respectively Mulligan and Keep)

Hypothesis : 

 * You are on the play
 * You have 7 cards in hand
 * You don't know the deck your opponent is playing

For Now only the Modern Boros Burn Deck, the Modern 5C Humans Deck and the Modern Bant Spirit Deck have been done

ETA on the quality of Mulligans :

  * Burn : Ok
  * 5C humans : Needs more data
  * Bant Spirit : Ok (93% on training set / 91% on testing set)

# Dependencies

You need to have the following packages installed in order to have this project work :

 * numpy
 * Pillow
 * matplotlib
 * pandas
 * sklearn
 * jupyter
 
You can always install them with pip (type : "pip install NAME_OF_THE_PACKAGE" in your terminal) 

# Run a hand

To run a hand use the following :

```
#Imports the magic.py file:
from magic import *

#If you want BantSpirit : (type 'Burn' for Burn and 'Humans' for Humans)
#Moreover if your computer is a bit old, you can try to add : Resolution ='low' to lower the resolution of the cards displayed
#It would give : mulliganer = Main('BantSpirit',Resolution = 'low')
mulliganer = Main('BantSpirit')

#Then write the hand you want to test (here it's spirit):
hand = 'horizon_canopy moorland_haunt cavern_of_souls  mausoleum_wanderer mausoleum_wanderer supreme_phantom collected_company'

#Test your hand ! 
mulliganer.RunHand(hand)
```

# worth noting ! 

The parameter "TrainingSetSize" has been set to 0 in every cell of the notebook in order not to have any picture in the notebook that are extremely heavy in MB. Try 10 to try for instance !
For the same reason, the RunHand methods have been commented but you can uncomment them to run them and check that everything works correctly !
