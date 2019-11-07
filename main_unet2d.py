import numpy as np
import scipy.io as sio
import tensorflow as tf
from train import Trainer
from image_loder import ImageLoder
from unet_2d import Model
import time
import platform

if platform.system() == 'Windows':
    root_path = '.'
if platform.system() == 'Linux':
    root_path = '.'

lr = 0.001
train_set = '10'
#test_set = str(300 - int(train_set))
test_set = '5'
data_path = root_path
result_path = root_path + '/result_norm_' + train_set + '_' + test_set + '_' + str(lr)

# enable eager execution
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

res = True
weights = ['feedback']
start = time.time()
for weight_type in weights:

    print(data_path + '/train_' + train_set + '/*.tif')
    dataset_train = ImageLoder(search_path=data_path + '/train_' + train_set + '/*.tif',
                               data_suffix='org',
                               label_suffix='lab',
                               n_class=3,
                               shuffle_data=True)
    print(data_path + '/validation/*.tif')
    dataset_validation = ImageLoder(search_path=data_path + '/validation/*.tif',
                                    data_suffix='org',
                                    label_suffix='lab',
                                    n_class=3,
                                    shuffle_data=False)


    data_provider = {'train': dataset_train, 'validation': dataset_validation}

    result_name = 'U'
    if res:
        result_name = result_name + 'R'
    if weight_type is not None:
        result_name = result_name + '_' + weight_type
    save_path = result_path + '/' + result_name

    unet2d = Model(n_class=3, n_layer=5,
                   features_root=16, filter_size=3, pool_size=2,
                   keep_prob=0.85,
                   weight_type=weight_type,
                   concat_or_add='concat',
                   res=res)

    batch_size = 2
    train_size = dataset_train.size()
    assert train_size % batch_size == 0
    iters = train_size // batch_size
    #test_data, test_y, test_mask = dataset_train(dataset_train.size())
    print('dataset size: %d, batch size: %d, iters: %d' % (train_size, batch_size, iters))

    trainer = Trainer(unet2d, learning_rate=lr, training_iters=iters, batch_size=batch_size)
    train_dice, validation_dice = trainer.train(data_provider, epochs=500, restore=False, output_path=save_path,
                                                train_summary=True, validation_summary=True)

    result_mat = {
        'train_dice': train_dice,

        'validation_dice': validation_dice}
    sio.savemat(save_path + '/train_result', result_mat)

    # test
    dataset_test = ImageLoder(search_path=data_path + '/test_' + test_set + '/*.tif',
                              data_suffix='org',
                              label_suffix='lab',
                              n_class=3,
                              shuffle_data=False)

    best_checkpoint = save_path + '/checkpoint/best_checkpoint'
    evaluation = trainer.evaluate(dataset_test, 1, best_checkpoint)
    with open(result_path + '/test_result.txt', 'a+') as f:
        f.write('%s\n' % result_name)
        for key in evaluation:
            value = evaluation.get(key)
            if value.size <= 1:
                f.write('%s: %s\n' % (key, value))
            else:
                f.write('%s\t' % key)
                for v in value:
                    f.write('%s\t' % v)
                f.write('\n')
        f.write('\n\n')

    dice_test = trainer.dice_eval(dataset_test, best_checkpoint)
    result_mat = {result_name + '_' + train_set + '_dice': dice_test}
    sio.savemat(result_path + '/dice_' + result_name, result_mat)

    test_data, test_y, test_mask = dataset_test(dataset_test.size())

    resPred = trainer.pred_image(test_data, best_checkpoint, dataset_test.size())
    sio.savemat(result_path + '/pred_' + result_name, {'Prediction':resPred})

print((time.time() - start))