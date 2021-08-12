
Semantic CMR Synthesis:

To train the style and morphological synthesis: 

1. Download the hdf5 dataset file, the SPADE checkpoints and the final SPADE model from https://mega.nz/file/NdJ3FQAT#gXjITYynLMS303zGMVwDurs0nb6MK4ck2MfTdY3rOnI
2. Extract original_dataset.hdf5 into  /datasets/myops_ds
3. Run  prepare_myops_SPADE.py, located in the same folder.
This will prepare the dataset for the regular SPADE training and augment it with warping augmentations. Each sample will be roi-cropped using the ground truth with a margin of 10 pixels and contrast-stretched. At test time for the final predictions, each sample will be cropped using a vanilla U-net trained for the joint scar,edema and myocardial predictions and contrast-stretched.
4. run train.py, 
5. Additionally, you can find the final checkpoints for our solution in the folder label2myops_256_vae_crop_norm, uncompress them in the checkpoints/label2myops_256_vae_crop_norm folder.

To reproduce the style transfer and morphological augmentations:
1. Uncompress spade_generator.myops in the main folder.
2. Run online_augmenation_myops.py to generate and visualize the different augmentation techniques and generate the augmented dataset.


![Alt text](example_synth.png?raw=true "Title")



BCDU-Net:

You can find the BCDU-Net official repositories here:
https://github.com/rezazad68/BCDU-Net 

Train it with the SPADE augmentations.





