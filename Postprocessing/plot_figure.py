import nibabel as nib
import matplotlib.pyplot as plt
import os
import pdb
name = str(123)
dir_post = 'D:\\Xenia\\Myops_challenge\\Results\\5models\\Training\\'
dir_pred = 'D:\\Xenia\\Myops_challenge\\Dataset\\preds_val\\preds_val\\'
dir_input = 'D:\\Xenia\\Myops_challenge\\Dataset\\preds_val\\training\\'
dir_out = 'D:\\Xenia\\Myops_challenge\\Results\\Visualization\\'
# Original
gt_nib = nib.load(os.path.join(dir_input, 'myops_training_'+ name +'_gd.nii.gz')).get_fdata()
co = nib.load(os.path.join(dir_input, 'myops_training_' + name + '_C0.nii.gz')).get_fdata()
t2 = nib.load(os.path.join(dir_input, 'myops_training_' + name + '_T2.nii.gz')).get_fdata()
de = nib.load(os.path.join(dir_input, 'myops_training_'+ name + '_DE.nii.gz')).get_fdata()
# Predicted by net
pred_myo_nib = nib.load(os.path.join(dir_pred, 'myops_training_' + name +'_pred_MY.nii.gz')).get_fdata()
pred_nib = nib.load(os.path.join(dir_pred, 'myops_training_' + name +'_pred_SE.nii.gz')).get_fdata()
# Post-processed
post_myo_nib = nib.load(os.path.join(dir_post, 'myo_myops_training_' + name +'.nii.gz')).get_fdata()
post_nib = nib.load(os.path.join(dir_post, 'myops_training_' + name +'_seg.nii.gz')).get_fdata()

plt.imshow(co[:, :, 1])
plt.axis('off')
plt.savefig(os.path.join(dir_out, "CO.png"), bbox_inches='tight')

plt.imshow(t2[:, :, 1])
plt.axis('off')
plt.savefig(os.path.join(dir_out, "T2.png"), bbox_inches='tight')

plt.imshow(de[:, :, 1])
plt.axis('off')
plt.savefig(os.path.join(dir_out, "de.png"), bbox_inches='tight')

myo_label = 200
edema_label = 1220
scar_label = 2221
gt_nib[gt_nib == 600] =0;
gt_nib[gt_nib == 500] =0;
plt.imshow(gt_nib[:, :, 1])
plt.axis('off')
plt.savefig(os.path.join(dir_out, "gt.png"), bbox_inches='tight')


myo_label = 200
edema_label = 1220
scar_label = 2221
pred_myo_nib[pred_myo_nib >0] = 200
pred_myo_nib[pred_nib == edema_label] = 1220
pred_myo_nib[pred_nib == scar_label] = 2221
plt.imshow(pred_myo_nib[:, :, 1])
plt.axis('off')
plt.savefig(os.path.join(dir_out, "pred.png"), bbox_inches='tight')


post_myo_nib[post_myo_nib >0] = 200
post_myo_nib[post_nib == edema_label] = 1220
post_myo_nib[post_nib == scar_label] = 2221
plt.imshow(post_myo_nib[:, :, 1])
plt.axis('off')
plt.savefig(os.path.join(dir_out, "post.png"), bbox_inches='tight')
pdb.set_trace()
