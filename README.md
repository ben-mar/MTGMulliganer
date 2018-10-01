# MTGMulliganer
Here is the Magic The Gathering Mulliganer, the Machine Learning algorithm that depending on the deck you've selected, tells you if you should keep or Mulligan your hand

For Now only the Modern Boros Burn Deck, the Modern 5C Humans Deck and the Modern Bant Spirit Deck have been done

ETA on the quality of Mulligans :

  * Burn : Ok
  * 5C humans : Needs more data
  * Bant Spirit : Needs more data
  
To run a hand use the following :

```
#Imports the magic.py file:
from magic import *

#If you want BantSpirit:
mulliganer = Main('BantSpirit')

#or if you want Burn:
mulliganer = Main('Burn') 

#or if you want Humans:
mulliganer = Main('Humans') 

#Then write the hand you want to test (here it's spirit):
hand = [' horizon_canopy moorland_haunt cavern_of_souls  mausoleum_wanderer mausoleum_wanderer supreme_phantom collected_company ']

#Test your hand ! 
mulliganer.RunHand(hand)
```

# worth noting ! 

The parameter "TrainingSetSize" has been set to 0 in every cell of the notebook in order not to have any picture in the notebook that are extremely heavy in MB. Try 10 to try for instance !
