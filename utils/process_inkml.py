import numpy as np
import pandas as pd
import imageio


def read_inkml(inkml):
    def parse(inkml):
        """
        read inkml file and return 1. The latex code of the equation, 2. The
        :param inkml: The file location and name of that inkml file
        :return: 1. Full Latex Code, 2. The coordinate of the pen, 3.(Not planned) Pairs of tracedata & the latex code.
        """
        latex = ''
        image_coord = []
        with open(inkml, 'r', errors = 'ignore') as ink:
            first = True
            idx = 0
            min_x = 1e6
            max_x = 0
            min_y = 1e6
            max_y = 0
            for line in ink:
                if line.find('<annotation type="truth">') != -1 and first:
                    line = line.partition('<annotation type="truth">')[-1]
                    line = line.partition('</annotation>')[0]
                    latex = line
                    first = False

                if line.find('<trace id="{}">'.format(idx)) != -1:
                    if line.find('</trace>') != -1:
                        line = line.partition('<trace id="{}">'.format(idx))[-1]
                        line = line.partition('</trace>')[0]
                    else:
                        line = ink.readline()
                    # print("Idx: ", idx)
                    # print(line)
                    idx = idx+1
                    points = line.split(', ')
                    # coord : [y, x]
                    coord = np.zeros((len(points), 2))
                    for i, p in enumerate(points):
                        
                        x = float(p.split()[0])
                        y = float(p.split()[1])
                        # put y coordinate first
                        coord[i, :] = [y, x]
                        if min_x > x:
                            min_x = x
                        if max_x < x:
                            max_x = x
                        if min_y > y:
                            min_y = y
                        if max_y < y:
                            max_y = y
                    image_coord.append(coord)
        minmax = (min_x, max_x, min_y, max_y)
        return latex, image_coord, minmax

    def drawing(coord, minmax, frame=(80, 448)):
        """
        From the data with point coordinates, rescale it to fit in the frame, maintaining the width/height ratio.
        :param coord: The list of np arrays, each np arrays contains the coordinates of the points of the pen.
        :param minmax: minimum, maximum value of x, y coordinates in inkml format, it is used for rescaling.
        :param frame: The size(h, w) that we want to fit the image in. Default is (80, 448)
        :return: 2D numpy array with size (h, w), Background is black and the letters are white.
        """
        h, w = frame
        min_x, max_x, min_y, max_y = minmax
        len_x = max_x - min_x
        len_y = max_y - min_y

        scale = min((w-1)/len_x, (h-1)/len_y)
        
        # red_x, red_y : used for centering the image in the frame.(redundant values, at least one should be zero)
        red_x = (w - scale*len_x)//2
        red_y = (h - scale*len_y)//2

        image = np.zeros((h, w))
        for stroke in coord:
            stroke[:, 0] = np.around((stroke[:, 0] - min_y) * scale)
            stroke[:, 1] = np.around((stroke[:, 1] - min_x) * scale)

            for i in range(stroke.shape[0]):
                image[int(red_y + stroke[i, 0]), int(red_x + stroke[i, 1])] = 255
                # Filling only two points is not sufficient, it is too sparse
                # Need to draw the interpolation coordinate between two consecutive points
                if i != 0:
                    for j in range(1, 10):
                        interp = np.around((j * stroke[i-1, :] + (10-j) * stroke[i, :])/10)
                        image[int(red_y + interp[0]), int(red_x + interp[1])] = 255

        return image

    latex, a, b = parse(inkml)
    image = drawing(a, b)
    image = image.astype('uint8')
    return latex, image


if __name__ == '__main__':
    latex_labels = []
    image_paths = []
    for i in range(1, 9139):
        if i % 500 == 0:
            print(i)
        filepath = 'inkml_dataset/' + str(i) + '.inkml'
        a, b = read_inkml(filepath)
        imagepath = 'inkml_images/' + str(i) + '.png'
        latex_labels.append(a)
        image_paths.append(imagepath)
        imageio.imsave('inkml_images/'+str(i) + '.png', b)
    # a, b = read_inkml('CROHME_training_2011/1.inkml')
    dataframe = pd.DataFrame({'image_paths': image_paths, 'latex_labels': latex_labels})
    dataframe.to_csv('dataset_inkml.csv')
    print("Done!")