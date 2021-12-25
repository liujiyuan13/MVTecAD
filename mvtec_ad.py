'''
This is written by Jiyuan Liu, Dec. 24, 2021.
Homepage: https://liujiyuan13.github.io.
Email: liujiyuan13@163.com.
All rights reserved.
'''

import os
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

from typing import Optional, Callable, List, Tuple, Dict, Any


class MVTecAD(VisionDataset):
    """
    `MVTec Anomaly Detection <https://www.mvtec.com/company/research/datasets/mvtec-ad/>`_ Dataset.
    In this class, dataset refers to mvtec-ad, while subset refers to the sub dataset, such as bottle.
    Args:
        root (string): Root directory of the MVTec AD Dataset.
        subset_name (string, optional): One of the MVTec AD Dataset names.
        train (bool, optional): If true, use the train dataset, otherwise the test dataset.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        mask_transform (callable, optional): A function/transform that takes in the
            mask and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
     Attributes:
        subset_name (str): name of the loaded subset.
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        data (list): List of PIL images. not named with 'images' for consistence with common dataset, such as cifar.
        masks (list): List of PIL masks. mask is of the same size of image and indicate the anomaly pixels.
        targets (list): The class_index value for each image in the dataset.
    Note:
        The normal class index is 0.
        The abnormal class indexes are assigned 1 or higher alphabetically.
    """

    # urls from https://www.mvtec.com/company/research/datasets/mvtec-ad/
    data_dict = {
        'mvtec_anomaly_detection': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'
    }
    subset_dict = {
        'bottle': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz',
        'cable': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz',
        'capsule': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz',
        'carpet': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz',
        'grid': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz',
        'hazelnut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz',
        'leather': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz',
        'metal_nut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz',
        'pill': 'https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz',
        'screw': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz',
        'tile': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz',
        'toothbrush': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz',
        'transistor': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz',
        'wood': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz',
        'zipper': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz'
    }

    # supported image extensions
    image_exts = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    # string definition specified to MVTec-AD dataset
    dataset_name = next(iter(data_dict.keys()))
    subset_names = list(subset_dict.keys())
    normal_str = 'good'
    mask_str = 'ground_truth'
    train_str = 'train'
    test_str = 'test'
    compress_ext = '.tar.xz'

    def __init__(self,
                 root,
                 subset_name: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 mask_transform: Optional[Callable] = None,
                 download=True):

        super(MVTecAD, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train
        self.mask_transform = mask_transform

        # path
        self.dataset_root = os.path.join(self.root, self.dataset_name)
        self.subset_name = subset_name.lower()
        self.subset_root = os.path.join(self.dataset_root, self.subset_name)
        self.subset_split = os.path.join(self.subset_root, self.train_str if self.train else self.test_str)

        if download is True:
            self.download()

        if not os.path.exists(self.subset_root):
            raise FileNotFoundError('subset {} is not found, please set download=True to download it.')

        # get image classes and corresponding targets
        self.classes, self.class_to_idx = self._find_classes(self.subset_split)

        # get images, masks and targets
        self.data, self.masks, self.targets = self._load_data(self.subset_split, self.class_to_idx, self.image_exts)
        if self.__len__() == 0:
            raise FileNotFoundError("found 0 files in {}\n".format(self.subset_split))

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        '''
        get item iter.
        :param idx (int): idx
        :return: (tuple): (image, mask, target) where target is index of the target class.
        '''
        # get image, mask and target of idx
        image, mask, target = self.data[idx], self.masks[idx], self.targets[idx]

        # apply transform
        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.transform(mask)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, mask, target

    def __len__(self) -> int:
        return len(self.targets)

    def extra_repr(self):
        split = self.train_str if self.train else self.test_str
        return 'using data: {data}\nsplit: {split}'.format(data=self.subset_name, split=split)

    def download(self):
        '''
        download the subset
        :return:
        '''
        os.makedirs(self.dataset_root, exist_ok=True)

        if os.path.exists(self.subset_root):
            return

        if self.subset_name not in self.subset_names:
            raise ValueError('The dataset called {} is not exist.'.format(self.subset_name))

        # download
        filename = self.subset_name + self.compress_ext
        download_and_extract_archive(self.subset_dict[self.subset_name], self.dataset_root, filename=filename)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.remove(self.normal_str)
        classes = [self.normal_str] + classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _load_data(self,
                   folder: str,
                   class_to_idx: Dict[str, int],
                   image_exts: Optional[Tuple[str, ...]]) -> Tuple[Any, Any, Any]:
        '''
        load images, masks and corresponding targets.
        :param folder: folder/class_0/*.*
                       folder/class_1/*.*
        :param class_to_idx: dict of class name and corresponding label
        :param image_exts: file extensions, such as .png
        :return: (images, masks, targets), they are np.array()
        '''
        # data to load
        data, masks, targets = [], [], []

        # load data
        for target_class in class_to_idx.keys():
            class_idx = class_to_idx[target_class]
            target_folder = os.path.join(folder, target_class)
            for root, _, fnames in sorted(os.walk(target_folder, followlinks=True)):
                for fname in fnames:
                    fext = '.' + fname.split('.')[-1]
                    if fext in image_exts:
                        # get image
                        image_path = os.path.join(root, fname)
                        data.append(Image.open(image_path))
                        # get mask
                        if target_class is self.normal_str:
                            masks.append(Image.new('L', data[-1].size))
                        else:
                            # only test data have mask images
                            mask_path = image_path.replace(self.test_str, self.mask_str)
                            mask_path = mask_path.replace(fext, '_mask'+fext)
                            masks.append(Image.open(mask_path))
                        # get target
                        targets.append(class_idx)

        # # transform to numpy.Array
        # data, masks = np.stack(data, axis=0), np.stack(masks, axis=0)
        # targets = np.array(targets)

        return data, masks, targets
