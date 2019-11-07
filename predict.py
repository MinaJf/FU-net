import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

import util as U

class Predictor:
    def __init__(self, model, checkpoint_path):
        self.model = model

        # self.learning_rate = tfe.Variable(0.001)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.global_step = tf.train.get_or_create_global_step()

        # restore
        checkpoint = tfe.Checkpoint(model=self.model.net)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
        

    def predict(self, xs, batch_size):
        xs_size = xs.shape[0]
        q = xs_size // batch_size
        r = xs_size % batch_size
        idx = 0
        ys = None
        while q != 0 or r != 0:
            if q != 0:
                sub_size = batch_size
                q -= 1
            elif r != 0:
                sub_size = r
                r = 0 

            sub_ys = self.model.predict(xs[idx:idx+sub_size])
            if ys is None:
                ys = sub_ys
            else:
                ys = np.concatenate((ys, sub_ys), axis=0)
            idx += sub_size
            print(idx)

        return np.array(ys)
    
    def eval(self, dataset_test, batch_size):
        v_size = dataset_test.size()

        evaluation = None

        q = v_size // batch_size
        r = v_size % batch_size

        while q != 0 or r != 0:
            if q != 0:
                sub_size = batch_size
                q -= 1
            elif r != 0:
                sub_size = r
                r = 0    
            sub_x, sub_y, sub_mask = dataset_test(sub_size)
            feed_dict = {'x':sub_x, 'y':sub_y, 'mask':sub_mask}
            sub_eval = self.model.evaluation(feed_dict)
            sub_pred = sub_eval.pop('prediction')
            evaluation = U.add_dict(evaluation, U.multiply_dict(sub_eval, float(sub_size) / v_size))
           
            
        # print
        output_str = U.eval_to_str(evaluation)
        print('validation %s'%output_str)
        return evaluation
