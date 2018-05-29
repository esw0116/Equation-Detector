# Equation-Detector
Equation Detector(For CV Term Project)

## Goal
TO detect a Equation from an image and export the result as a Latex form

**Difference from other Detector**

It detects the form directly from the image, not the inkML.
(InkML contains the order of the drawing.)

## Plan
1. Make a single Symbol Detector using several kinds of Networks.
(Caution : We should unify the image size in order to inference the class correctly)

2. After the performance of the single symbol classification is verified, 
put the equation image to the model and compare the feature.

3. Use LSTM to train the order of the symbols.

Default input image size 96 * 480 \
INKML images are reshaped to 80 * 448, and will be put in the center of the frame.

## TODO
- [x] Edit loss plot(need to reduce bouncing graph)
- [x] Encode the latex code according to the dictionary
- [ ] Implement loading model from pt file
- [ ] Add resume functions(Prerequisite : Save optimizer params)
- [ ] Distinguish CAPITAL letters from small letters in symbol directory
- [ ] Initialize the parameter(Inception)
- [x] Check the encoding algorithm by making a decoder
- [ ] Keep updating dictionary, check for errors
- [ ] Create Dataloader for RNN input/output