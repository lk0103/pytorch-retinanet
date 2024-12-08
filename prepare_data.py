from scipy.io import loadmat
import xml.etree.ElementTree as ET
import random


def extract_annotations(file_path):
    global classes_list
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize a list to store annotations
    annotations = []

    # Iterate over all objects in the file
    for obj in root.findall('.//object'):
        # Extract bounding box
        bndbox = obj.find('bndbox')
        x1 = bndbox.find('xmin').text
        y1 = bndbox.find('ymin').text
        x2 = bndbox.find('xmax').text
        y2 = bndbox.find('ymax').text

        # Extract class name
        class_name = obj.find('name').text
        classes_list.add(class_name)

        # Add to the list as a tuple
        annotations.append([x1, y1, x2, y2, class_name])

    return annotations

def csv_file_annotations(img_type, csv_file_name):
    lines = []
    print(len(img_type))
    for file in img_type:
        file_path = './stanford_dogs/Images/' + file
        file = file.replace('.jpg', '')
        annotation_path = './stanford_dogs/Annotation/' + file
        annotations = extract_annotations(annotation_path)
        for annotation in annotations:
            csv_data = [file_path, *annotation]
            lines.append(','.join(csv_data))
    with open(csv_file_name, 'w') as f:
        f.write('\n'.join(lines))

def csv_class_list(classes, csv_file_name):
    lines = []
    print(len(classes))
    for i, class_name in enumerate(classes):
        csv_data = [class_name, str(i)]
        lines.append(','.join(csv_data))
    with open(csv_file_name, 'w') as f:
        f.write('\n'.join(lines))

train_list = loadmat('./stanford_dogs/train_list.mat')
test_list = loadmat('./stanford_dogs/test_list.mat')

train_images = train_list['file_list']
test_images = test_list['file_list']


train_images = [item[0][0] for item in train_images]
test_images = [item[0][0] for item in test_images]

random.shuffle(train_images)
val_images = train_images[:2058]
train_images = train_images[2058:]

classes_list = set()
#9942 - 48.31% of the whole dataset - 10656 dogs
csv_file_annotations(train_images, 'train_annots.csv')
#2058 - 10% of the whole dataset - 2221 dogs
csv_file_annotations(val_images, 'val_annots.csv')
#8580 - 41.69% of the whole dataset - 9249 dogs
csv_file_annotations(test_images, 'test_annots.csv')
random.shuffle(test_images)
csv_file_annotations(test_images[:30], 'test_annots_smaller.csv')

#120 classes
csv_class_list(classes_list, 'class_list.csv')


#commands
#python train.py --dataset csv --csv_train train_annots.csv  --csv_classes class_list.csv  --csv_val val_annots.csv --depth 18 --epochs 4
#python csv_validation.py --csv_annotations_path val_annots.csv --model_path model_final.pt --images_path ./ --class_list_path class_list.csv --iou_threshold 0.5
#python csv_validation.py --csv_annotations_path test_annots.csv --model_path model_final.pt --images_path ./ --class_list_path class_list.csv --iou_threshold 0.5
#python visualize.py --dataset csv --csv_classes class_list.csv  --csv_val test_annots_smaller.csv --model model_final.pt



