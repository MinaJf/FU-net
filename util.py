import numpy as np
from PIL import Image

def recale_array(array, nmin=None, nmax=None):
    array = np.array(array)
    if nmin is None:
        nmin = np.min(array)
    array = array - nmin
    if nmax is None:
        nmax = np.max(array) + 1e-10
    array = array / nmax
    array = (array * 255).astype(np.uint8)

    return array


def combine_images(x, y, pred, mask=None):
    pred = np.array(pred)
    if pred.ndim == 4:
        return combine_2d_images(x, y, pred)
    elif pred.ndim == 5:
        return combine_3d_images(x, y, mask, pred)
    else:
        raise 'Unknow dimensions of ouput image!'

def combine_2d_images(x, y, pred):

    n_class = pred.shape[-1]
    n_y = pred.shape[2]
    x_img = recale_array(x)

    imgs = []
    # result for classes
    for i in range(n_class):
        img = np.concatenate((x_img.reshape(-1, n_y),
                              recale_array(y[..., i]).reshape(-1, n_y),
                              recale_array(pred[..., i]).reshape(-1, n_y)),
                             axis=1)
        imgs.append(img)

    # result for argmax
    pred_max = np.argmax(pred, axis=-1)
    y_max = np.argmax(y, axis=-1)
    img = np.concatenate((x_img.reshape(-1, n_y),
                          recale_array(y_max, nmin=0, nmax=n_class-1).reshape(-1, n_y),
                          recale_array(pred_max, nmin=0, nmax=n_class-1).reshape(-1, n_y)),
                         axis=1)
    imgs.append(img)

    return imgs



def combine_3d_images(x, y, mask, pred):
    n_class = pred.shape[-1]
    n_y = pred.shape[2]
    n_z = pred.shape[3]
    x_img = recale_array(x)

    imgs = []
    # result for classes
    for i in range(n_class):
        img = np.concatenate((x_img.transpose((0,1,3,2,4)).reshape(-1, n_y, order='F'),
                              recale_array(y[..., i]).transpose((0,1,3,2)).reshape(-1, n_y, order='F'),
                              recale_array(mask[..., i]).transpose((0,1,3,2)).reshape(-1, n_y, order='F'),
                              recale_array(pred[..., i]).transpose((0,1,3,2)).reshape(-1, n_y, order='F')),
                             axis=1)
        imgs.append(img)

    # result for argmax
    pred_max = np.argmax(pred, axis=-1)
    y_max = np.argmax(y, axis=-1)
    img = np.concatenate((x_img.transpose((0,1,3,2,4)).reshape(-1, n_y, order='F'),
                              recale_array(y_max).transpose((0,1,3,2)).reshape(-1, n_y, order='F'),
                              recale_array(mask[...,0]).transpose((0,1,3,2)).reshape(-1, n_y, order='F'),
                              recale_array(pred_max).transpose((0,1,3,2)).reshape(-1, n_y, order='F')),
                             axis=1)
    imgs.append(img)

    return imgs

def save_images(imgs, path):
    for i in range(len(imgs)-1):
        Image.fromarray(imgs[i]).save('%s_class_%d.png'%(path, i))
    Image.fromarray(imgs[-1]).save('%s_argmax.png'%path)

def combine_and_save_images(x, y, pred, path):
    imgs = combine_images(x, y, pred)
    save_images(imgs, path)


def eval_to_str(evaluation_dict):
    o_s = ''
    for key in evaluation_dict:
        value = evaluation_dict.get(key)
        if value.size >= 3:
            mean = np.mean(value[1:])
        else:
            mean = np.mean(value)
        o_s += '%s: %.4f  '%(key, mean)
    return o_s

def add_dict(dict_old, dict_new):
    if dict_old is None:
        dict_old = dict_new
    else:
        for key in dict_new:
            dict_old[key] += dict_new[key]
    return dict_old
    
def div_dict(_dict, i):
    for key in _dict:
            _dict[key] /= i
    return _dict

def multiply_dict(_dict, i):
    for key in _dict:
            _dict[key] *= i
    return _dict
