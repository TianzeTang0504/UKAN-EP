import h5py
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
modalities = ('t1c', 't1n', 't2f', 't2w',)

train_set = {
        'root': "C:\\Users\\TTZ\\Desktop\\val",
        'out': "D:\\brats2024\\data\\valdata",
        'flist': "D:\\brats2024\\unet\\val.txt",
        }

def process_h5(path, out_path):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    label = sitk.GetArrayFromImage(sitk.ReadImage(path + 'seg.nii.gz')).transpose(1, 2, 0)
    #print(label.shape)
    #4 x (H,W,D) -> (4,H,W,D)
    images = np.stack([sitk.GetArrayFromImage(sitk.ReadImage(path + modal + '.nii.gz')).transpose(1, 2, 0) for modal in modalities], 0)  # [240,240,155]
    label = label.astype(np.uint8)
    images = images.astype(np.float32)
    #case_name = path.split('/')[-1] #linux
    case_name = os.path.split(path)[-1]  # windows
    case_name = case_name[10:]
    
    outpath = os.path.join(out_path, case_name)
    output = outpath + 'mri_norm2.h5'
    mask = images.sum(0) > 0
    for k in range(4):

        x = images[k,...]  #
        y = x[mask]

        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[k,...] = x
    #print(case_name, images.shape, label.shape)
    print(case_name, images.shape)
    f = h5py.File(output, 'w')
    f.create_dataset('image', data=images, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()


def doit(dset):
    root, out_path = dset['root'], dset['out']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    names = ['BraTS-GLI-' + sub for sub in subjects]
    paths = [os.path.join(root, name, name + '-') for name in names]

    for path in tqdm(paths):
        process_h5(path, out_path)
        # break
    print('Finished')


if __name__ == '__main__':
    doit(train_set)
