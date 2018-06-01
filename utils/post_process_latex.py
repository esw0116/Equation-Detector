import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from option import args
from utils.check_encoding import check_encoding

def post_process(args):
    characters = args.dictionary
    print(characters)

    data_csv = pd.read_csv('../Dataset/dataset_inkml.csv')
    latex_labels = data_csv['latex_labels']
    latex_labels_np = latex_labels.values
    image_paths = data_csv['image_paths']

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
            while index != -1:
                if index == -2:
                    index = labels_encoded.find(character, 0)
                else:
                    index = labels_encoded.find(character, index+1)
                if index != -1:
                    indices.append([index, i, char_length])

        removed.append(labels_removed)
        # encoded.append(labels_encoded)
        indices_np = np.array(indices)
        indices_np = indices_np[np.lexsort((indices_np[:, 1], indices_np[:, 0]))]
        # remove redundancies
        for idx, char_id, char_len in indices_np:
            if char_len > 1:
                for i in range(0, char_len):
                    delete_rows = np.where(indices_np[:, 0] == idx+i)[0]
                    for delete in delete_rows:
                        if indices_np[delete, 1] != char_id:
                            indices_np[delete] = [-1, -1, -1]

        indices_np = indices_np[indices_np[:, 0] >= 0]

        encoded_values = indices_np[:, 1]
        encoded.append(encoded_values)

    dataframe = pd.DataFrame({'encoded': encoded, 'original': original, 'image_paths': image_paths})
    dataframe.to_csv('../Dataset/encoded_dataset.csv')

    #TODO: REMOVE REDUNDANCIES!

if __name__ == "__main__":
    #post_process(args)
    check_encoding(args)