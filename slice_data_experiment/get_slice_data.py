import os

import argparse
import pathlib
import numpy as np

import matplotlib.pyplot as plt

from datasets import get_datasets_brats


def main():
    parser = argparse.ArgumentParser(description='Get Brats 2019 slice data.')
    parser.add_argument('--data_root', default='MICCAI_BraTS_2019_Data_Training/HGG_LGG', help='data directory')
    parser.add_argument('--output_root', default='MICCAI_BraTS_2019_Data_Training_Slice/HGG_LGG', help='data output directory')
    args = parser.parse_args()
    save_folder = pathlib.Path(args.output_root)
    save_folder.mkdir(parents=True, exist_ok=True)
    train_dataset, _ = get_datasets_brats(data_root=args.data_root)

    slice_data = list()

    i = 0
    for volume_data in train_dataset:
        slice_data_dir = os.path.join(str(save_folder), volume_data['patient_id'])
        print(slice_data_dir)
        
        volume_image, volume_label = volume_data['image'].numpy(), volume_data['label'].numpy()
        print(volume_image.shape)
        slice_sum = np.sum(volume_label, (0, 2, 3))
        valid_slice = np.where(slice_sum>0)

        valid_slice_image = volume_image[:, valid_slice].squeeze(axis=1)
        valid_slice_label = volume_label[:, valid_slice].squeeze(axis=1)

        print(valid_slice_image.shape)
        print(valid_slice_label.shape)

        num_slice = valid_slice_image.shape[1]

        for slice_idx in range(num_slice):
            slice_data.append(
                dict(
                    patient_id=volume_data["patient_id"],
                    image=valid_slice_image[:, slice_idx],
                    label=valid_slice_label[:, slice_idx],)
            )

        i += 1
        if i == 2:
            break


    print(slice_data[0]['image'][0].shape)
    plt.figure()
    plt.imshow(slice_data[0]['image'][0])
    plt.figure()
    plt.imshow(slice_data[0]['label'][0])
    plt.show()
    


if __name__ == '__main__':
    main()