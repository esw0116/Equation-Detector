import numpy as np
import pandas as pd

'''
CODE TO CHECK WHETHER ENCODING WAS DONE CORRECTLY OR NOT
'''


def check_parse():
    data_csv = pd.read_csv('../Dataset/regenerated_dataset.csv')
    regenerated = data_csv['regenerated']
    # regenerated = regenerated.values
    original = data_csv['original']
    # original = original.values

    truth_count = 0
    false_count = 0

    count = 0
    for regen, orig in zip(regenerated, original):
        regen = regen.replace(" ", "")
        orig = orig.replace(" ", "")
        if regen == orig:
            truth_count += 1
        else:
            for regen_letter, original_letter in zip(regen, orig):
                if regen_letter != original_letter:
                    print("At count: ", count)
                    print("R: ", regen)
                    print("O: ", orig)
                    print("Regen letter: {}, Original: {}".format(regen_letter, original_letter))
                    false_count += 1

        count += 1
    print("Truth: ", truth_count)
    print("False: ", false_count)


if __name__ == "__main__":
    check_parse()