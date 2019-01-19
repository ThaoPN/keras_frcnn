import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import sys


def parse_element(element):
    class_name = element.find('name').text
    bndbox = element.find('bndbox')
    x1 = bndbox.find('xmin').text
    x2 = bndbox.find('xmax').text
    y1 = bndbox.find('ymin').text
    y2 = bndbox.find('ymax').text
    
    return x1, y1, x2, y2, class_name

def get_voc_data(input_path):
    """Parse the data from annotation file

    Args:
        input_path: annotation file path

    Returns:
        all_data: list(filepath, width, height, list(bboxes))
        classes_count: dict{key:class_name, value:count_num} 
            e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True
    
    images_dir = os.path.join(input_path, 'images')
    xmls_dir = os.path.join(input_path, 'labels')
    
    xmls = [f for f in os.listdir(xmls_dir) if os.path.isfile(os.path.join(xmls_dir, f))]
    
    annotation = []
    for xml in xmls:
        if xml.startswith('.'):
            continue
        xml_path = os.path.join(xmls_dir, xml)
        # print(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        #print(xml)
        image_name = xml.replace('.xml', '.jpg')
        image_file = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_file):
            continue
            
        for element in root.iter('object'):
            x1, y1, x2, y2, class_name = parse_element(element)
            if class_name == 'person':
                annotation.append((image_name, x1, y1, x2, y2, class_name))

    i = 1


    print('Parsing annotation files')

    for line in annotation:

        # Print process
        sys.stdout.write('\r'+'idx=' + str(i))
        i += 1

        # Make sure the info saved in annotation file matching the format (path_filename, x1, y1, x2, y2, class_name)
        # Note:
        #	One path_filename might has several classes (class_name)
        #	x1, y1, x2, y2 are the pixel value of the origial image, not the ratio value
        #	(x1, y1) top left coordinates; (x2, y2) bottom right coordinates
        #   x1,y1-------------------
        #	|						|
        #	|						|
        #	|						|
        #	|						|
        #	---------------------x2,y2

        (filename,x1,y1,x2,y2,class_name) = line

        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

        if class_name not in class_mapping:
            if class_name == 'bg' and found_bg == False:
                print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                found_bg = True
            class_mapping[class_name] = len(class_mapping)

        if filename not in all_imgs:
            all_imgs[filename] = {}
            file_path = os.path.join(images_dir, filename)
            img = cv2.imread(file_path)
            (rows,cols) = img.shape[:2]
            all_imgs[filename]['filepath'] = file_path
            all_imgs[filename]['width'] = cols
            all_imgs[filename]['height'] = rows
            all_imgs[filename]['bboxes'] = []
            # if np.random.randint(0,6) > 0:
            # 	all_imgs[filename]['imageset'] = 'trainval'
            # else:
            # 	all_imgs[filename]['imageset'] = 'test'

        all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])

    # make sure the bg class is last in the list
    if found_bg:
        if class_mapping['bg'] != len(class_mapping) - 1:
            key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
            val_to_switch = class_mapping['bg']
            class_mapping['bg'] = len(class_mapping) - 1
            class_mapping[key_to_switch] = val_to_switch

    return all_data, classes_count, class_mapping

def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and not found_bg:
                    print('Found class name with special name bg. Will be treated as a'
                          ' background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)),
                 'y2': int(float(y2))})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping

