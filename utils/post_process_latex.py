import numpy as np
import pandas as pd

special_characters = ['(', ')', 'sum', '_', '{', '}', '=', '^', 'frac', '+', '-', 'cos', 'sin',
            'sqrt', 'forall', 'in', '!', 'cdots', 'int', 
             'geq', 'neq', 'infty', '.', 'log', 'tan', '&', '[', ']', '|', 'forall', 'times', 'div', 'ldots', 'pm',
             'arrow', 'lim', 'cdot', 'leq', 'lt']
greek_characters = ['theta', 'pi', 'mu', 'sigma', 'lambda','beta', 'gamma', 'alpha', 'Delta', 'phi']
numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#temporary
special_characters_temp = ['(', ')', '\sum', '_', '{', '}', '=', '^', r'\frac', '+', '-', '\cos', '\sin',
            '\sqrt', r'\forall', '\in', '!', '\cdots', '\int',
             '\geq', r'\neq', '\infty', '.', '\log', r'\tan', '&', '[', ']', '|', r'\forall', r'\times', '\div', '\ldots', '\pm']
greek_characters_temp = [r'\theta', '\pi', '\mu', '\sigma', '\lambda', r'\beta', '\gamma', r'\alpha', '\Delta', '\phi']

# strings already omitted
omit = ['Bigg', 'Big', r'\left', r'\right', r'\mbox']

characters = []
characters.extend(special_characters)
characters.extend(greek_characters)
characters.extend(numbers)
characters.extend(letters)

# sort and reverse
characters = sorted(characters, key=len)
characters = list(reversed(characters))             ## <-------- LEXICON

data_csv = pd.read_csv('dataset_inkml_1.csv')
latex_labels = data_csv['latex_labels']
latex_labels_np = latex_labels.values

removed = []
encoded = []
original = []
for labels in latex_labels_np:
    labels_removed = labels
    labels_encoded = labels
    original.append(labels)
    indices = []
    
    for i, character in enumerate(characters):
        index = -2  # -1 should only be exit code
        char_length = len(character)
        labels_removed = labels_removed.replace(character, "")
        # labels_encoded = labels_encoded.replace(character, " #" + str(i)+ "# ")
        while index!=-1:
            if index ==-2:
                index = labels_encoded.find(character, 0)
            else:
                index = labels_encoded.find(character, index+1)
            if index != -1:
                indices.append([index, i, char_length])


    removed.append(labels_removed)
    # encoded.append(labels_encoded)
    indices_np = np.array(indices)
    indices_np = indices_np[np.lexsort((indices_np[:,1], indices_np[:,0]))]
    #remove redundancies
    for idx, char_id, char_len in indices_np:
        if char_len > 1:
            for i in range(0, char_len):
                delete_rows = np.where(indices_np[:,0]==idx+i)[0]
                for delete in delete_rows:
                    if indices_np[delete, 1] != char_id:
                        indices_np[delete] = [-1, -1, -1]

    indices_np = indices_np[indices_np[:,0]>=0]

    encoded_values = indices_np[:,1]
    encoded.append(encoded_values)

     


dataframe = pd.DataFrame({'removed': removed, 'encoded': encoded, 'original': original})

dataframe.to_csv('encoded_dataset.csv')

#TODO REMOVE REDUNDANCIES!