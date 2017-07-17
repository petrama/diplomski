import os

import numpy as np

import xml.etree.cElementTree as ET



import matplotlib.pyplot as plt
import argparse




def read_bboxes(cityscapes_root,subset):
    imagesets_path=os.path.join(cityscapes_root,'image_sets',subset+'.txt')

    with open(imagesets_path) as f:
        file_names=f.readlines()





    ann_path = os.path.join(cityscapes_root, 'object_annotations',subset)

    aspects=np.array([])
    scales=np.array([])
    for j,line in enumerate(file_names):
        line=line.strip()
        annotation=os.path.join(ann_path,line+'_bboxes.xml')
        tree = ET.parse(annotation)
        xmins = tree.findall('object/bndbox/xmin')
        xmins=np.array([int(x.text) for x in xmins])
        xmax=tree.findall('object/bndbox/xmax')
        xmax=np.array([int(x.text) for x in xmax])
        width=xmax-xmins

        ymins=tree.findall('object/bndbox/ymin')
        ymins=np.array([int(x.text) for x in ymins])

        ymax=tree.findall('object/bndbox/ymax')
        ymax=np.array([int(x.text) for x in ymax])


        height=ymax-ymins
        #print(width)

        #if (width==0).any():
        #    print(line)
        #    print(width,height)

        scale=width*height#*1.0/(2048*1024)
        sizes=width*height
        if scales.any():
            scales=np.hstack([scales,scale])
        else:
            scales=scale

        aspect_ratio=height*1.0/width
        if aspects.any():
            aspects=np.hstack([aspects,aspect_ratio])
        else:
            aspects=aspect_ratio
        #print(width,height)


    #print(np.histogram(scales,bins=10,range=[0.0,1]))
    return scales,aspects


def plot_hist_aspects(a,s):






    # the histogram of the data
    n, bins, patches = plt.hist(a_filtered, bins=10, range=(0, 5), alpha=0.5)

    print(n, bins)
    # add a 'best fit' line
    # y = mlab.normpdf( bins, mu, sigma)
    # l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.title('Aspect ratio histogram 1')
    plt.axis([0, 4, 0, 60000])
    plt.grid(True)

    plt.savefig('histogram_aspect_where_scale<0.025.png')





def plot_hist_scales(s):


    # the histogram of the data

    max=20000
    n, bins, patches = plt.hist(s, bins=20, range=(0, max), alpha=0.5)
    print('matplotlib')
    print(n, bins)
    # add a 'best fit' line
    # y = mlab.normpdf( bins, mu, sigma)
    # l = plt.plot(bins, y, 'r--', linewidth=1)

    print('Kumulativna suma:',np.cumsum(n))

    plt.title('Histogram površina oznake okvira (veličine objekta)')
    plt.axis([0, max, 0, 20000])
    plt.grid(True)

    plt.savefig('histogram_objects_scale.png')








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cityscapes_root', help="Folder with fine_annotations and leftImg8bit",default='/home/pmarce/datasets/cityscapes')
    parser.add_argument('--subset', help="train or val, subfolders",default='train')
    args = parser.parse_args()

    s,a=read_bboxes(args.cityscapes_root,  args.subset)


    plot_hist_scales(s)
    #plot_hist_aspects(a,s)


