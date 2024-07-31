import os

# 定义数据路径
train_data_path = "D:\\brats2024\\data\\dataset"
#val_data_path = "D:\\brats2024\\data\\dataset"

train_ids = os.listdir(train_data_path)
#val_ids = os.listdir(val_data_path)


#Here, change the file name range by yourself
def process_filename(filename):
    return filename[:19]

processed_train_ids = [process_filename(file) for file in train_ids]
#processed_val_ids = [process_filename(file) for file in val_ids]

processed_train_ids.sort()
#processed_val_ids.sort()

output_dir = "D:\\brats2024\\unet"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(processed_train_ids))

#with open(os.path.join(output_dir, 'valid.txt'), 'w') as f:
    #f.write('\n'.join(processed_val_ids))

print("success write {}/train.txt and {}/valid.txt".format(output_dir, output_dir))