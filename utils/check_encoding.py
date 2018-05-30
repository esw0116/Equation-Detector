import numpy as np
import pandas as pd


numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#temporary
special_characters_temp = ['(', ')', '\sum', '_', '{', '}', '=', '^', r'\frac', '+', '-', '\cos', '\sin',
            '\sqrt', r'\forall', '\in', '!', '\cdots', '\int',
             '\geq', r'\neq', '\infty', '.', '\log', r'\tan', '&', '[', ']', '|', r'\forall', r'\times', '\div', '\ldots', '\pm',
             '\cdot', r'\arrow', '\lim', '\leq', '\lt']
greek_characters_temp = [r'\theta', '\pi', '\mu', '\sigma', '\lambda', r'\beta', '\gamma', r'\alpha', '\Delta', '\phi']

characters = []
characters.extend(special_characters_temp)
characters.extend(greek_characters_temp)
characters.extend(numbers)
characters.extend(letters)

# sort and reverse
characters = sorted(characters, key=len)
characters = list(reversed(characters))             ## <-------- LEXICON in correct order

''' 
USING ENCODED DATA TO REGENERATE LATEX STRING
COMMENT WHEN NOT NEEDED
'''

data_csv = pd.read_csv('encoded_dataset.csv')
encoded = data_csv['encoded']
encoded = encoded.values
original = data_csv['original']

regen_list = []

for encoding in encoded:
    # get rid of weird symbols
    encoding = encoding.replace("[", "")
    encoding = encoding.replace("]", "")
    encoding = encoding.replace("\n", "")
    regenerated = ""
    encoding = encoding.split(" ")
    for num in encoding:
        if not (num == "" or num== " "):
            regenerated = regenerated +" " + characters[int(num)]
            
    regen_list.append(regenerated)    
    print(regenerated)

dataframe = pd.DataFrame({'regenerated': regen_list, 'encoded':encoded, 'original':original})

dataframe.to_csv('regenerated_dataset.csv')