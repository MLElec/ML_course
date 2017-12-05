
import numpy as np
import datetime
import Augmentor as ag
import os
import shutil
from os import listdir
from os.path import isfile, join
from os import rename
from PIL import Image


# returns modification date
def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)

# sorts the file in modification time order and renames them in this way
def rename_sort_files(prefix ,mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    # get the creation time
    dates = {}
    for file in onlyfiles:
        dates[file] = modification_date(os.path.join(mypath,file))
    
    # rename according to the creation time
    i=1
    for filename, date in sorted(dates.items(), key=lambda item: (item[1], item[0])):
        im = Image.open(os.path.join(mypath,filename))
        im.convert('L') # convert to monochrome
        print('Renaming '+ filename + ' to ' +  prefix+'{:03d}'.format(i) + '.png')
        im.save(os.path.join(mypath,prefix +'{:03d}'.format(i) + '.png') )
        os.remove(os.path.join(mypath,filename))
        i += 1


# cretaes rotated and flipped road data, saves it in an "output" folder
def create_new_road_data(nr_new_data, path_labels, path_images, prefix_name='satImage_', prob =0.3 , seed=100):
    # remove previous data from 'output' folder
    shutil.rmtree(path_labels + '/output')
    shutil.rmtree(path_images + '/output')

    # generate twice the same data (the same because of the same seed)
    p = ag.Pipeline();
    ag.Pipeline.set_seed(seed)
    p.add_further_directory(path_images) # ensure you press enter after this, don't just c&p this code.
    p.flip_left_right(probability=prob)
    p.flip_top_bottom(probability=prob)
    p.rotate(probability=prob, max_left_rotation=25, max_right_rotation=25)

    # save the data
    p.sample(nr_new_data)

    p = ag.Pipeline();
    ag.Pipeline.set_seed(seed)
    p.add_further_directory(path_labels) # ensure you press enter after this, don't just c&p this code.

    ag.Pipeline.set_seed(seed)
    p.flip_left_right(probability=prob)
    p.flip_top_bottom(probability=prob)
    p.rotate(probability=prob, max_left_rotation=25, max_right_rotation=25)
    # save the data
    p.sample(nr_new_data)


    # label the data the same way
    rename_sort_files(prefix_name, path_labels +'/output');
    rename_sort_files(prefix_name, path_images +'/output');