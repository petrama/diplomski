import os
import numpy as np
import xml.etree.cElementTree as ET

def create_list_per_clas(cityscapes_root, subset, all_labels):
    imagesets_path = os.path.join(cityscapes_root, 'image_sets', subset + '.txt')

    with open(imagesets_path) as f:
        file_names = f.readlines()

    matrix = np.ones((len(all_labels), len(file_names)))

    ann_path = os.path.join(cityscapes_root, 'object_annotations', subset)

    for j, line in enumerate(file_names):
        line = line.strip()
        annotation = os.path.join(ann_path, line + '_bboxes.xml')
        tree = ET.parse(annotation)
        names = tree.findall('object/name')
        names = set(n.text for n in names)
        for i, label in enumerate(all_labels):
            if label not in names:
                matrix[i, j] = -1


    for row,label in enumerate(all_labels):
        file_lines=[' '.join((a.strip(),str(int(b)))) for a,b in zip(file_names,matrix[row])]
        file_content='\n'.join(file_lines)
        file_list_name='{}_{}.txt'.format(label,subset)
        file_list_path=os.path.join(cityscapes_root,'image_sets',file_list_name)

        with open(file_list_path,'+w') as f:
            f.write(file_content)
            f.flush()

        print('created file: {}'.format(file_list_name))




    print(names)
    #print(matrix)


if __name__=='__main__':

    all_labels=['person',
                  'rider',
                  'car',
                  'truck',
                  'bus',
                  'caravan',
                  'trailer',
                  'train',
                  'motorcycle',
                  'bicycle']
    create_list_per_clas('/home/pmarce/datasets/cityscapes','val',all_labels)