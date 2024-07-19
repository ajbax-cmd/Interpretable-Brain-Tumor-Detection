import os
import shutil
# Script to rename image and label files to numerically sequential matching names
def organize_dataset(root_dir, output_dir, mode='Training'):
    class_dirs = [d for d in os.listdir(os.path.join(root_dir, mode)) if os.path.isdir(os.path.join(root_dir, mode, d))]
    image_output_dir = os.path.join(output_dir, mode.lower(), 'images')
    label_output_dir = os.path.join(output_dir, mode.lower(), 'labels')
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    count = 0
    for class_label in class_dirs:
        class_dir = os.path.join(root_dir, mode, class_label)
        for image_filename in os.listdir(class_dir):
            if image_filename.endswith(('.jpg', '.jpeg', '.png')):
                src_image_path = os.path.join(class_dir, image_filename)
                new_image_name = f'y{count}.jpg'
                dst_image_path = os.path.join(image_output_dir, new_image_name)
                shutil.copyfile(src_image_path, dst_image_path)
                
                label_filename = f'y{count}.txt'
                label_file_path = os.path.join(label_output_dir, label_filename)
                with open(label_file_path, 'w') as label_file:
                    label_file.write(class_label)
                
                count += 1

def create_data_yaml(output_dir):
    data_yaml_content = f"""
train: {output_dir}/training/images
val: {output_dir}/validation/images
test: {output_dir}/testing/images
nc: 4
names: ['class 0', 'class 1', 'class 2', 'class 3']
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as yaml_file:
        yaml_file.write(data_yaml_content)



def main():
    root_dir = '/home/alan/Documents/YOLOV8_interpretable/Brain_Tumor_MRI'  # Change this to your dataset path
    output_dir = '/home/alan/Documents/YOLOV8_interpretable/Dataset_3'  # Change this to your desired output path
    
    organize_dataset(root_dir, output_dir, mode='Training')
    organize_dataset(root_dir, output_dir, mode='Testing')
    create_data_yaml(output_dir)

if __name__ == '__main__':
    main()
