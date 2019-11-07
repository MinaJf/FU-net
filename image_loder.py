import glob
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import scipy.ndimage as nd

class ImageLoder:
    def __init__(self, data_suffix, label_suffix, n_class, search_path=None, filename_list=None, mask_suffix=None, shuffle_data=True, resize=None):
        self.data_suffix = data_suffix
        self.label_suffix = label_suffix
        self.shuffle_data = shuffle_data
        self.mask_suffix = mask_suffix
        self.n_class = n_class
        self.resize = resize

        assert (search_path is None) ^ (filename_list is None), 'Multiple data list input!'
        if search_path is not None:
            self.filename_list = self._search_files(search_path)
        else:
            self.filename_list = filename_list
        assert len(self.filename_list) > 0, 'No data!'
        print('%d file loaded'%len(self.filename_list))
        self.file_idx = len(self.filename_list)

    def size(self):
        return len(self.filename_list)

    def ndim(self):
        return 4

    def __call__(self, n_data):

        imgs, labs, masks = self._load_and_process_data()
        
        for i in range(1, n_data):
            img, lab, mask = self._load_and_process_data()
            imgs = np.concatenate((imgs, img), axis=0)
            labs = np.concatenate((labs, lab), axis=0)
            if masks is not None:
                masks = np.concatenate((masks, mask), axis=0)

        return imgs, labs, masks

    def _load_and_process_data(self):
        img, lab, mask = self._next_data()

        img, lab, mask = self._pre_process(img, lab, mask)

        # assert img.shape[0:-1] == lab.shape[0:-1]
        img = img.reshape([1] + list(img.shape) + [1])
        lab = lab.reshape([1] + list(lab.shape))

        if mask is not None:
            mask = mask.reshape([1] + list(mask.shape))
            assert lab.shape == mask.shape

        return img, lab, mask

    def _pre_process(self, img, lab, mask):

        img = self._process_img(img)
        lab = self._process_lab(lab)
        mask = self._process_mask(mask)

        if self.resize is not None:
            img = cv2.resize(img, self.resize)
            lab = cv2.resize(lab, self.resize)
            if mask is not None:
                mask = cv2.resize(mask, self.resize)
        
        return img, lab, mask

    def _process_img(self, img):

        # standardization (zero mean)
        img -= np.mean(img)
        img /= np.std(img)

        # 
        img -= np.min(img)
        img /= np.max(img)

        return img

    def _process_lab(self, lab):
        nx = lab.shape[0]
        ny = lab.shape[1]
        labs = np.zeros((nx, ny, self.n_class), dtype=np.float32)

        for i in range(self.n_class):
            labs[..., i][lab==i] = 1
        return labs

    def _process_mask(self, mask):
        if mask is not None:
            mask = np.stack((mask,mask,mask), 3)
        return mask

    def _search_files(self, search_path):
        files = glob.glob(search_path)
        return [name for name in files if self.data_suffix in name and not self.label_suffix in name]

    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)

    def _cycle_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.filename_list):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.filename_list)

    def _next_data(self):
        self._cycle_file()
        image_name = self.filename_list[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.label_suffix)
        img = self._load_file(image_name)
        lab = self._load_file(label_name)

        if self.mask_suffix is None:
            return img, lab, None
        else:
            mask_name = image_name.replace(self.data_suffix, self.mask_suffix)
            mask = self._load_file(mask_name)
        return img, lab, mask
