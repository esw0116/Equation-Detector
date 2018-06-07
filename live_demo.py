import torch
import numpy as np
import cv2
from model import EncoderCNN
from model import CRNNv2
from option import args
from data import common


device = torch.device('cpu')
encoder = EncoderCNN.make_model(args)
encoder.eval()
decoder = CRNNv2.make_model(args)
decoder.eval()

encoder.load_state_dict(torch.load('./Inference_model/model_enc_best.pt', map_location='cpu'))
decoder.load_state_dict(torch.load('./Inference_model/model_best.pt', map_location='cpu'))
print("Model Loaded")

#read image
img = np.zeros((1,96,480))
img_read = cv2.imread('trial.png',0)
img_read = cv2.resize(img_read, (480, 96), interpolation = cv2.INTER_NEAREST)
img_read[img_read>100]=255
cv2.imwrite('input_img.png',img_read)

img_read = common.normalize_img(img_read)
print(img_read.shape)
img_read_expanded = np.expand_dims(img_read, axis=0)
img = img_read_expanded
img = np.expand_dims(img, axis=0)

# Input image into network
image_tensor = torch.from_numpy(img)
image_tensor = image_tensor.to(torch.float).to(device)
with torch.autograd.no_grad():
    features = encoder(image_tensor)
    output = decoder.sample(features)

equation = output.cpu().numpy().squeeze(0)
latex_str = ""

for i in equation:
    latex_str += args.dictionary[int(i)]

print(latex_str)

