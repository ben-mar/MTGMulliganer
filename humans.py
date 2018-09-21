# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:31:26 2018

@author: B3K
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 19:18:22 2018

@author: Benji

This file regroups evry function used for the magic_v2 Keep or Mull on Boros Burn
"""
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
#from sklearn.ensemble import RandomForestClassifier

deck_dict =    {"island":1,
                "plains":1,
                "seachrome_coast":1,
                "unclaimed_territory":4,
                "ancient_ziggurat":4,
                "cavern_of_souls":4,
                "horizon_canopy":4,
                "aether_vial":4,
                "noble_hierarch":4,
                "champion_of_the_parish":4,
                "kitesail_freebooter":4,
                "meddling_mage":4,
                "phantasmal_image":4,
                "thalia_guardian_of_thraben":3,
                "thalia_lieutenant":4,
                "mantis_rider":4,
                "militia_bugler":3,
                "reflector_mage":3}

# Here is the card list 
card_list = ["island",
            "plains",
            "seachrome_coast",
            "unclaimed_territory",
            "ancient_ziggurat",
            "cavern_of_souls",
            "horizon_canopy",
            "aether_vial",
            "noble_hierarch",
            "champion_of_the_parish",
            "kitesail_freebooter",
            "meddling_mage",
            "phantasmal_image",
            "thalia_guardian_of_thraben",
            "thalia_lieutenant",
            "mantis_rider",
            "militia_bugler",
            "reflector_mage"]

features_raw = np.copy(pd.read_csv('Training_set_humans/Features.csv',sep=';'))[:,1:]

def make_encoding(list_from_docs,docs):
    encoding_list = [[] for line in docs]
    i=0
    for line in docs:
        line = line.split()
        for word in line:
            encoding_list[i] += [list_from_docs.index(word.lower())]
        i+=1
    return encoding_list

def decklist(deck_dict):
    deck_list =[]
    cards = deck_dict.items()
    for card_itself,number_of_the_card in cards:
        for _ in range(number_of_the_card):
            deck_list.append(str(card_itself))
    return deck_list

def cards_to_index(deck_list):
    card_index = []
    deck_list_set = list(set(deck_list))
    for card in deck_list:
        card_index.append(deck_list_set.index(card))
    return card_index,deck_list_set

def create_hand(deck_list,n):
    deck_numbers_index,deck_list_map = cards_to_index(deck_list)
    deck_numbers_shuffled = [i for i in deck_numbers_index]
    np.random.shuffle(deck_numbers_shuffled)
    hand_numbers = deck_numbers_shuffled[0:n]
    hand_names = [deck_list_map[i] for i in hand_numbers]
    return hand_names  

def created_hand_to_testable_hand(created_hand):
    """
    takes the output of create_hand function as an input : turns ['card1','card2', ...] into ['card1 card2 ...'] 
    """
    testable_hand = ''
    for card in created_hand:
        testable_hand += card + ' '
    return([testable_hand])

def sort_hand(hand,card_list):
    hand_to_sort = [card_list.index(hand[i].lower()) for i in range(len(hand))]
    hand_to_sort.sort()
    hand_sorted = [card_list[i] for i in hand_to_sort]
    return (hand_sorted)

def show_hand(testable_hand):
    hand = testable_hand[0]
    list_im =[]
    for word in hand.split():
        list_im.append('images_Humans/'+word.lower()+'.jpg')
    imgs    = [ Image.open(i) for i in list_im ]
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    plt.figure(figsize=(20, 16), dpi= 100, facecolor='w', edgecolor='k')
    plt.imshow(imgs_comb)
    plt.axis('off')
    plt.show()  

def testable_hand_for_RFC(testable_hand):
    testable_hand = sort_hand(testable_hand[0].split(),card_list)
    testable_hand_RFC = []
    for i in range(len(testable_hand)):
        converted_card = convert_card_into_features(testable_hand[i],features_raw)
        testable_hand_RFC = np.concatenate((testable_hand_RFC,converted_card))
    return([testable_hand_RFC])
        


#model_magic_RFC = RandomForestClassifier(n_estimators=50,max_depth=10, random_state=0)
#model_magic_RFC.fit(X, y)

def Test_a_testable_hand(testable_hand,model_RFC): 
    testable_hand_RFC = testable_hand_for_RFC(testable_hand)
    if model_RFC.predict(testable_hand_RFC)[0] ==1:
        print('-------------------------------------------------------KEEP-------------------------------------------------------','\n')
        print('\n')
        print('\n')
    else :
        print('-------------------------------------------------------MULL-------------------------------------------------------','\n')
        print('\n')
        print('\n')

def Run_a_hand(hand,model_RFC):
    sorted_hand = sort_hand(hand[0].split(),card_list)
    testable_hand = created_hand_to_testable_hand(sorted_hand)
    show_hand(testable_hand)
    Test_a_testable_hand(testable_hand,model_RFC)

def Test_model(deck_dict,n_test,model_RFC):
    deck_list = decklist(deck_dict)
    for _ in range(n_test):  
        created_hand = create_hand(deck_list,7)
        testable_hand = created_hand_to_testable_hand(created_hand)
        Run_a_hand(testable_hand,model_RFC)

def training_set(deck_dict,n,training_set_size,training_file_name):
    pv =';'
    with open(training_file_name+'.csv' , 'w') as training_file,\
    open(training_file_name+'question.csv' , 'w') as question_file:
        deck_list = decklist(deck_dict)
        for _ in range(training_set_size):
            current_hand = create_hand(deck_list,n)
            sorted_hand = sort_hand(current_hand,card_list)
            testable_sorted_hand = created_hand_to_testable_hand(sorted_hand)         
            show_hand(testable_sorted_hand)           
            y = input("Keep: 1, Mull: 0 or Not sure: 3 ?")
            print('\n')
            line_written=''
            for card in sorted_hand:
                line_written += card+pv
            line_written += y + '\n'
            y=int(y)
            if y==1 or y==0:
                training_file.write(line_written)
            if y==3:
                question_file.write(line_written)
            if y==9:
                break 
    print('Written !')

def training_set_to_docs_and_labels(training_file_name):
    training_set = pd.read_csv(training_file_name+'.csv',sep =';',header=None)
    training_set = np.copy(training_set)
    docs = []
    for line in training_set:
        line = line[0:7]
        docs.append(" ".join(line))
    labels = training_set[:,7]           
    return(docs,labels)
    
def convert_card_into_features(card,features_raw):
    index = card_list.index(card)
    feature = features_raw[index]
    return(feature)

def new_features_from_docs(docs,labels,training_file_name):
    pv =';'
    with open(training_file_name+'.csv' , 'w') as training_file:
        for i in range(len(docs)):
            docs_i = sort_hand(docs[i].split(),card_list)
            line_to_write=''
            for j in range(7):
                feature_card = convert_card_into_features(docs_i[j],features_raw)
                K = len(feature_card)
                for k in range(K):
                    line_to_write += str(int(feature_card[k]))+pv
            line_to_write+=str(labels[i])
            training_file.write(line_to_write+'\n')
    print('Done !')  