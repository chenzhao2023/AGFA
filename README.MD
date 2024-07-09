This is the code for the paper "AGFA-Net: Attention-Guided Feature-Aggregated Network for Coronary Artery Segmentation using Computed Tomography Angiography"

Dataset: CCTA dataset



Explanation for the code:

1， 01_File_process.py.  What is this file doing: First, rename the original 80G data to a folder name and file name. Then move all the files to the specified folder, and finally move the labeled files to the trainMask folder.

2. 01_saveDataToTrain.py  converts the data of train and trainMask into dataset format (.pth file) and saves it in 5 parts.

3. train_AGFANet.py  Train the model using train-CASnet_XYbest_batch128.py, which corresponds to csnet3d in the model.

4. test_AGFANet.py   Test model, results for one person, subject level.

5. Visual3D_AGFANet.ipynb      View a 3D segmentation image of a person.







































