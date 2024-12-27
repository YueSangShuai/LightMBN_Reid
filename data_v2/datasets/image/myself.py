from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re
import warnings

from .. import ImageDataset


class MyselfDateset(ImageDataset):
    def __init__(self,root='', **kwargs):
        dataset_dir = 'myselfdateset'
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, dataset_dir)

        self.data_dir = self.dataset_dir
        if not osp.isdir(self.data_dir):
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')
        
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'test')
        self.gallery_dir = osp.join(self.data_dir, 'test')



        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]

        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)


        super(MyselfDateset, self).__init__(train, query, gallery, **kwargs)

    def get_ids(self,dir_path):
        # 获取目录中的所有条目（文件和文件夹）
        return [item for item in os.listdir(dir_path) 
                if os.path.isdir(os.path.join(dir_path, item))]
    
    
    def get_filelist(self,dir):
        Filelist = []
        for home, dirs, files in os.walk(dir):
            for filename in files:
                # 文件名列表，包含完整路径

                Filelist.append(os.path.join(home, filename))

                # # 文件名列表，只包含文件名

                # Filelist.append( filename)

        return Filelist
    
    
    def process_dir(self, dir_path, relabel=False):
        pid_container=set(self.get_ids(dir_path))
        ####### modified #######
        pid2label = {pid:label for label, pid in enumerate(sorted(pid_container))}
        ########################

        data = []
        for folder,pid in pid2label.items():
            folder_iamge_files=self.get_filelist(os.path.join(dir_path,folder))
            data_period=[(image_path,pid,-1) for image_path in folder_iamge_files]
            data+=data_period
            
        return data