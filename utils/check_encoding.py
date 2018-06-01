import numpy as np
import pandas as pd

''' 
USING ENCODED DATA TO REGENERATE LATEX STRING
COMMENT WHEN NOT NEEDED
'''


def check_encoding(args):
    data_csv = pd.read_csv('../Dataset/encoded_dataset.csv')
    encoded = data_csv['encoded']
    encoded = encoded.values
    original = data_csv['original']
    characters = args.dictionary

    regen_list = []

    truth_count = 0
    false_count = 0
    max_len_encoding = 0
    max_count = 0

    for count, (encoding, orig) in enumerate(zip(encoded, original)):
        # get rid of weird symbols
        encoding = encoding.replace("[", "")
        encoding = encoding.replace("]", "")
        encoding = encoding.replace("\n", "")
        regen = ""
        encoding = encoding.split(" ")
        if len(encoding) > max_len_encoding:
            max_len_encoding = len(encoding)
            max_count = count
        for num in encoding:
            if not (num == "" or num == " "):
                regen = regen + " " + characters[int(num)]

        regen_list.append(regen)
        # print(regen)

        regen = regen.replace(" ", "")
        orig = orig.replace(" ", "")
        if regen == orig:
            truth_count += 1
        else:
            print("At count: ", count)
            print("R: ", regen, len(regen))
            print("O: ", orig, len(orig))
            false_count += 1
            for regen_letter, original_letter in zip(regen, orig):
                if regen_letter != original_letter:
                    print("Regen letter: {}, Original: {}".format(regen_letter, original_letter))

    print("Truth: ", truth_count)
    print("False: ", false_count)
    print("Max len: " , max_len_encoding)
    print("Max count: ", max_count)
    dataframe = pd.DataFrame({'regenerated': regen_list, 'encoded': encoded, 'original': original})
    dataframe.to_csv('../Dataset/regenerated_dataset.csv')
