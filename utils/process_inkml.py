import numpy as np
import pandas as pd
import imageio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from option import args
from utils.post_process_latex import post_process
from utils.check_encoding import check_encoding

omit = ['\Bigg', '\Big', '\left', r'\right', '\mbox']


def read_inkml(inkml):
    def parse(inkml):
        """
        read inkml file and return 1. The latex code of the equation, 2. The
        :param inkml: The file location and name of that inkml file
        :return: 1. Full Latex Code, 2. The coordinate of the pen, 3.(Not planned) Pairs of tracedata & the latex code.
        """
        latex = ''
        image_coord = []
        with open(inkml, 'r', errors='ignore') as ink:
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
                    l1dist = np.sum(np.abs(stroke[i-1, :] - stroke[i, :]))
                    interp_factor = int(l1dist // 2) + 5
                    for j in range(1, interp_factor):
                        interp = np.around((j * stroke[i-1, :] + (interp_factor-j) * stroke[i, :])/interp_factor)
                        image[int(red_y + interp[0]), int(red_x + interp[1])] = 255

        return image

    latex, a, b = parse(inkml)
    latex = latex.replace("\Bigg", "")
    latex = latex.replace("\Big", "")
    latex = latex.replace("\left", "")
    latex = latex.replace(r"\rightarrow", r"\to")
    latex = latex.replace(r"\right", "")
    latex = latex.replace("{}", "")
    for st in [r'\mbox {', r'\hbox {', r'\vtop {', '\mathrm {', r'\mbox{', r'\hbox{', r'\vtop{', '\mathrm{']:
        while True:
            cnt = 0
            idx = latex.find(st)
            if idx == -1:
                break

            latex = latex[:idx] + latex[idx + len(st):]
            while cnt >= 0:
                i1 = latex.find('{', idx)
                i2 = latex.find('}', idx)
                if i1 == -1:
                    i1 = len(latex)
                if i2 == -1:
                    i2 = len(latex)
                if i1 < i2:
                    cnt += 1
                    idx = i1 + 1
                else:
                    cnt -= 1
                    idx = i2 + 1
            latex = latex[:i2] + latex[i2 + 1:]

    # latex = latex.replace("\mbox", "")
    # latex = latex.replace(r"\hbox", "")
    # latex = latex.replace(r"\vtop", "")
    latex = latex.replace("\mathrm", "")
    latex = latex.replace(r"\dots", r"\cdots")
    latex = latex.replace(r"\cdots", r"\ldots")
    latex = latex.replace("$", "")
    latex = latex.replace("&gt;", ">")
    latex = latex.replace("\gt", ">")
    latex = latex.replace("&lt;", "<")
    latex = latex.replace("\lt", "<")
    latex = latex.replace("\lbrack", "[")
    latex = latex.replace(r"\rbrack", "]")
    latex = latex.replace("\n", "")
    image = drawing(a, b)
    image = image.astype('uint8')
    return latex, image


if __name__ == '__main__':
    latex_labels = []
    image_paths = []
    if not os.path.exists('Dataset/inkml_images'):
        os.mkdir('Dataset/inkml_images')
    for i in range(1, 13724):
        if i % 1000 == 0:
            print(i)
        fname = str('{:05}'.format(i))
        filepath = 'Dataset/inkml_dataset/' + fname + '.inkml'
        a, b = read_inkml(filepath)
        if a != "":
            imagepath = 'images/' + fname + '.png'
            latex_labels.append(a)
            image_paths.append(imagepath)
            imageio.imsave('Dataset/inkml_images/' + fname + '.png', b)
        else:
            print("None!!, {}".format(i))

    # a, b = read_inkml('CROHME_training_2011/1.inkml')
    dataframe = pd.DataFrame({'image_paths': image_paths, 'latex_labels': latex_labels})
    dataframe.to_csv('Dataset/dataset_inkml.csv')
    print("Done!")

    post_process(args)
    check_encoding(args)
