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
from sklearn.ensemble import RandomForestClassifier

deck_dict =    {"Wooded_Foothills" :4,
                "Stomping_Ground"  :1,
                "Sacred_Foundry"  :2,
                "Mountain" :3,
                "Inspiring_Vantage" :4,
                "Bloodstained_Mire" :4,
                "Arid_Mesa" :2,
                "Skullcrack" :4,
                "Searing_Blaze" :4,
                "Lightning_Helix" :4,
                "Lightning_Bolt" : 4,
                "Boros_Charm" :4,
                "Rift_Bolt" :4,
                "Lava_Spike" :4,
                "Monastery_Swiftspear" :4,
                "Goblin_Guide" :4,
                "Eidolon_of_the_Great_Revel" :4}

# Here is the card list 
card_list = ['inspiring_vantage', 'sacred_foundry', 'bloodstained_mire', 'wooded_foothills', 
    'arid_mesa', 'mountain', 'stomping_ground', 'monastery_swiftspear', 
    'goblin_guide', 'rift_bolt', 'lava_spike', 'lightning_bolt', 'searing_blaze',
    'eidolon_of_the_great_revel', 'lightning_helix','skullcrack', 'boros_charm']

creature_list = ['monastery_swiftspear', 'goblin_guide','eidolon_of_the_great_revel']

land_list = ['inspiring_vantage', 'sacred_foundry', 'bloodstained_mire', 'wooded_foothills', 
    'arid_mesa', 'mountain', 'stomping_ground']

CCM_1_list = ['monastery_swiftspear', 'goblin_guide', 'rift_bolt', 'lava_spike', 'lightning_bolt']

CCM_2_list = ['searing_blaze','eidolon_of_the_great_revel', 'lightning_helix','skullcrack', 'boros_charm']

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
        for i in range(number_of_the_card):
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
        list_im.append('images_Burn/'+word.lower()+'.jpg')
    imgs    = [ Image.open(i) for i in list_im ]
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    plt.figure(figsize=(20, 16), dpi= 100, facecolor='w', edgecolor='k')
    plt.imshow(imgs_comb)
    plt.axis('off')
    plt.show()  

def convert_cards_into_features(card,creature_list,land_list,CCM_2_list,CCM_1_list):
    """
    converts the card into a feature
    features are :
    
    * 1st feature ! a number representing the card (1st feature)
    * 2nd feature ! the CCM of the card 
    * 3rd feature ! the type of the card  : 
        0 if it's a land, 
        1 if a creature, 
        2 if a spell
    * 4th feature ! the type of color produced if it is a land :
        0 if it's not a land
        1 if it's red,
        2 if it's RW
    * 5th feature ! the color needed for the card :
        0 if it's a land,
        1 for red,
        2 for RW
    
    """
    
    card_list = ['inspiring_vantage', 'sacred_foundry', 'bloodstained_mire', 'wooded_foothills', 
    'arid_mesa', 'mountain', 'stomping_ground', 'monastery_swiftspear', 
    'goblin_guide', 'rift_bolt', 'lava_spike', 'lightning_bolt', 'searing_blaze',
    'eidolon_of_the_great_revel', 'lightning_helix','skullcrack', 'boros_charm']

    red_lands_only_list = ['mountain', 'stomping_ground']
    
    spell_list = set(card_list).difference(set(creature_list).union(land_list))

    spell_list_CCM_2_only_red = ['skullcrack','searing_blaze']
    feature = np.zeros((5,))
    
    card = card.lower()
    
    # feature 1
    feature[0]=card_list.index(card)
    
    if card in land_list:
        if card in red_lands_only_list:
            feature[1:]=[0,0,1,0]
        else :
            feature[1:]=[0,0,2,0]
    if card in creature_list:
        if card in CCM_1_list:
            feature[1:]=[1,1,0,1]
        else :
            feature[1:]=[2,1,0,1]
    if card in spell_list:
        if card in CCM_1_list:
            feature[1:]=[1,2,0,1]
        else :
            if card in spell_list_CCM_2_only_red:
                feature[1:]=[2,2,0,1]
            else :
                feature[1:]=[2,2,0,2]
    return(feature)

def testable_hand_for_RFC(testable_hand):
    testable_hand = sort_hand(testable_hand[0].split(),card_list)
    testable_hand_RFC = []
    for i in range(len(testable_hand)):
        converted_card = convert_cards_into_features(testable_hand[i],creature_list,land_list,CCM_2_list,CCM_1_list).reshape(1,5)[0]
        testable_hand_RFC = np.concatenate((testable_hand_RFC,converted_card))
    return([testable_hand_RFC])
        
data = np.copy(pd.read_csv('Training_set_burn/training_set.csv',sep=';',header=None))
print(data.shape)
X, y = data[:,:35],data[:,35]

model_magic_RFC = RandomForestClassifier(n_estimators=50,max_depth=10, random_state=0)
model_magic_RFC.fit(X, y)

def Test_a_testable_hand(testable_hand,model_RFC=model_magic_RFC): 
    testable_hand_RFC = testable_hand_for_RFC(testable_hand)
    if model_magic_RFC.predict(testable_hand_RFC)[0] ==1:
        print('-------------------------------------------------------KEEP-------------------------------------------------------','\n')
        print('\n')
        print('\n')
    else :
        print('-------------------------------------------------------MULL-------------------------------------------------------','\n')
        print('\n')
        print('\n')

def Run_a_hand(hand):
    sorted_hand = sort_hand(hand[0].split(),card_list)
    testable_hand = created_hand_to_testable_hand(sorted_hand)
    show_hand(testable_hand)
    Test_a_testable_hand(testable_hand)

def Test_model(deck_dict,n_test):
    deck_list = decklist(deck_dict)
    for i in range(n_test):  
        created_hand = create_hand(deck_list,7)
        testable_hand = created_hand_to_testable_hand(created_hand)
        Run_a_hand(testable_hand)

def training_set(deck_dict,n,training_set_size,training_file_name):
    pv =';'
    with open(training_file_name+'.csv' , 'w') as training_file:
        deck_list = decklist(deck_dict)
        for i in range(training_set_size):
            current_hand = create_hand(deck_list,n)
            sorted_hand = sort_hand(current_hand,card_list)
            testable_sorted_hand = created_hand_to_testable_hand(sorted_hand)         
            show_hand(testable_sorted_hand)           
            y = input("Keep : 1 or Mull:0 ?")
            print('\n')
            line_written=''
            for card in sorted_hand:
                line_written += card+pv
            line_written += y + '\n'
            training_file.write(line_written)
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
    
def new_features_from_docs(docs,labels,training_file_name):
    pv =';'
    with open(training_file_name+'.csv' , 'w') as training_file:
        for i in range(len(docs)):
            docs_i = sort_hand(docs[i].split(),card_list)
            line_to_write=''
            for j in range(7):
                feature_card = convert_cards_into_features(docs_i[j],creature_list,land_list,CCM_2_list,CCM_1_list)
                K = len(feature_card)
                for k in range(K):
                    line_to_write += str(int(feature_card[k]))+pv
            line_to_write+=str(labels[i])
            training_file.write(line_to_write+'\n')
    print('Done !')   