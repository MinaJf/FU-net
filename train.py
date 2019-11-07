from __future__ import absolute_import, division, print_function

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

import util as U

# from tf_eager._weight import balance_weight_map, feedback_weight_map, nan_to_zero

class Trainer:
    def __init__(self, model, learning_rate, training_iters, batch_size):
        self.model = model
        self.learning_rate = tfe.Variable(learning_rate)
        self.training_iters = training_iters
        self.batch_size = batch_size

        self.lr_step = 0
        def get_lr():
            if self.lr_step > 0 and self.lr_step % training_iters == 0:
                self.learning_rate.assign_sub(self.learning_rate * 0.005)
            self.lr_step += 1
            return self.learning_rate

        self.optimizer = tf.train.AdamOptimizer(learning_rate)

    def train_mini_batch(self, data_provider, epochs, mini_batch_size, output_path, restore=False, train_summary=True, validation_summary=True):

        # make dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        summary_path = output_path+'/summary'
        prediction_path = output_path+'/prediction'
        checkpoint_path = output_path+'/checkpoint'

        if not restore:
            shutil.rmtree(summary_path, ignore_errors=True)
            shutil.rmtree(prediction_path, ignore_errors=True)
            shutil.rmtree(checkpoint_path, ignore_errors=True)

        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # init global_step
        global_step = tf.train.get_or_create_global_step()
        global_step.assign(0)
        # restore
        checkpoint = tfe.Checkpoint(model=self.model.net, optimizer=self.optimizer, global_step=global_step, learning_rate=self.learning_rate)
        if restore:
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

        # summary writer
        train_writer = tf.contrib.summary.create_file_writer(summary_path+'/train')
        validation_writer = tf.contrib.summary.create_file_writer(summary_path+'/validation')

        dataset_train = data_provider.get('train')
        dataset_validation = data_provider.get('validation')

        train_auc = []
        validation_auc = []
        train_dice = []
        validation_dice = []
        max_gradient_norm = 5.0
        while global_step < epochs:
            train_evaluation = None
            for i in range(self.training_iters):
                assert self.batch_size % mini_batch_size == 0, 'batch_size % mini_batch_size != 0'
                mini_batch = self.batch_size // mini_batch_size
                grads = None
                cliped_grads = None
                for j in range(mini_batch):
                    mini_batch_x, mini_batch_y, mini_batch_mask = dataset_train(mini_batch_size)
                    feed_dict = {'x':mini_batch_x, 'y':mini_batch_y, 'mask':mini_batch_mask}
                    mini_grads = self.model.get_grads(feed_dict)

                    if grads is None:
                        grads = mini_grads
                    else:
                        for g_i, g in enumerate(mini_grads):
                            if g is not None:
                                grads[g_i] += g
                for g_i in range(len(grads)):
                    if grads[g_i] is not None:
                        grads[g_i] /= mini_batch
                        cliped_grads[g_i] = tf.clip_by_norm(grads, max_gradient_norm)


                self.optimizer.apply_gradients(zip(cliped_grads, self.model.net.variables))

                if train_summary:
                    sub_eval = self.model.evaluation(feed_dict)
                    sub_eval.pop('prediction')
                    train_evaluation = U.add_dict(train_evaluation, sub_eval)

                print('step: %d'%i)

            if train_summary:
                train_evaluation = U.div_dict(train_evaluation, self.training_iters)
                train_evaluation['learning rate'] = self.learning_rate.numpy()
                output_str = U.eval_to_str(train_evaluation)
                print('epoch %d -- train %s'%(global_step, output_str), end='| ')
                self.write_summary(train_evaluation, train_writer)

                train_auc.append(train_evaluation['auc'])
                train_dice.append(train_evaluation['dice'])

            summary, imgs = self._store_prediction(dataset_validation, '%s/epoch_%d'%(prediction_path, global_step))
            validation_auc.append(summary['auc'])
            validation_dice.append(summary['dice'])


            if validation_summary:
                self.write_summary(summary, validation_writer)
                self.write_image_summary(imgs, validation_writer)


            # save model
            if global_step.numpy() % 20 == 0:
                checkpoint.save(checkpoint_path+'/ckpt')

            global_step.assign_add(1)

        # finish
        checkpoint.save(checkpoint_path+'/ckpt')

        return train_auc, validation_auc, train_dice, validation_dice


    def train(self, data_provider, epochs, output_path, restore=False, train_summary=True, validation_summary=True):

        # make dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        summary_path = output_path+'/summary'
        prediction_path = output_path+'/prediction'
        checkpoint_path = output_path+'/checkpoint'

        if not restore:
            shutil.rmtree(summary_path, ignore_errors=True)
            shutil.rmtree(prediction_path, ignore_errors=True)
            shutil.rmtree(checkpoint_path, ignore_errors=True)

        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # init global_step
        global_step = tf.train.get_or_create_global_step()
        global_step.assign(0)
        # restore
        self.checkpoint = tfe.Checkpoint(model=self.model.net, optimizer=self.optimizer, global_step=global_step, learning_rate=self.learning_rate)
        if restore:
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

        # summary writer
        train_writer = tf.contrib.summary.create_file_writer(summary_path+'/train')
        validation_writer = tf.contrib.summary.create_file_writer(summary_path+'/validation')

        dataset_train = data_provider.get('train')
        dataset_validation = data_provider.get('validation')

        #train_auc = []
        #validation_auc = []
        train_dice = []
        validation_dice = []

        best_dice = -1

        while global_step < epochs:
            train_evaluation = None
            for i in range(self.training_iters):
                batch_x, batch_y, batch_mask = dataset_train(self.batch_size)
                feed_dict = {'x':batch_x, 'y':batch_y, 'mask':batch_mask}
                grads = self.model.get_grads(feed_dict)
                max_gradient_norm = 5.0
                #gradients = [
                #    None if gradient is None else tf.clip_by_norm(gradient, 5.0)
                #    for gradient in grads]
                clipped_gradients, self.gradient_norms = tf.clip_by_global_norm(grads, max_gradient_norm)
                self.optimizer.apply_gradients(zip(clipped_gradients, self.model.net.variables))

                if train_summary:
                    sub_eval = self.model.evaluation(feed_dict)
                    sub_eval.pop('prediction')
                    train_evaluation = U.add_dict(train_evaluation, sub_eval)

                print('step: %d'%i)
            print('epoch %d -- '%global_step, end='')
            if train_summary:
                train_evaluation = U.div_dict(train_evaluation, self.training_iters)
                train_evaluation['learning rate'] = self.learning_rate.numpy()
                output_str = U.eval_to_str(train_evaluation)
                print('train %s'%output_str, end='| ')
                self.write_summary(train_evaluation, train_writer)

                #train_auc.append(train_evaluation['auc'])
                train_dice.append(train_evaluation['dice'])

            summary, imgs = self._store_prediction(dataset_validation, '%s/epoch_%d'%(prediction_path, global_step))
            #validation_auc.append(summary['auc'])
            validation_dice.append(summary['dice'])

            mean_dice = np.mean(summary['dice'][1:2])
            if mean_dice > best_dice:
                best_dice = mean_dice
                self.checkpoint.write(checkpoint_path+'/best_checkpoint')

            if validation_summary:
                self.write_summary(summary, validation_writer)
                self.write_image_summary(imgs, validation_writer)


            # save model
            if global_step.numpy() % 50 == 0:
                self.checkpoint.save(checkpoint_path+'/ckpt')

            global_step.assign_add(1)

        # finish
       # self.checkpoint.save(checkpoint_path+'/ckpt')

        return  train_dice, validation_dice

    def pred_image(self, image_in, checkpoint_path, num):
        self.checkpoint = tfe.Checkpoint(model=self.model.net)
        self.checkpoint.restore(checkpoint_path)

        preds = []
        for icounter in range(num):
            im = image_in[icounter, ...]
            imNew = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])
            prob = np.array(self.model.predict(imNew))
            pred = np.argmax(prob, -1)
            preds.append(pred)

        return np.array(preds)



    def dice_eval(self, dataset_validation, checkpoint_path):
        self.checkpoint = tfe.Checkpoint(model=self.model.net)
        self.checkpoint.restore(checkpoint_path)
        v_size = dataset_validation.size()

        dice = []

        for i in range(v_size):
            sub_x, sub_y, sub_mask = dataset_validation(1)
            feed_dict = {'x':sub_x, 'y':sub_y, 'mask':sub_mask}
            sub_dice = self.model.evaluation_dice(feed_dict)
            dice.append(sub_dice)
		
        return np.array(dice)

    def evaluate(self, dataset_test, batch_size, checkpoint_path=None):
        if checkpoint_path is not None:
            self.checkpoint.restore(checkpoint_path)
        bk_batch_size = self.batch_size
        self.batch_size = batch_size
        evaluation, _ = self._store_prediction(dataset_test, None)
        self.batch_size = bk_batch_size
        return evaluation

    def _store_prediction(self, dataset_validation, path):
        ndim = dataset_validation.ndim()
        if ndim == 4:
            return self._store_prediction_2d(dataset_validation, path)
        elif ndim == 5:
            return self._store_prediction_3d(dataset_validation, path)
        else:
            raise 'Unknow dimensions of prediction!'


    def _store_prediction_2d(self, dataset_validation, path):
        v_size = dataset_validation.size()

        evaluation = None
        imgs = None

        q = v_size // self.batch_size
        r = v_size % self.batch_size

        while q != 0 or r != 0:
            if q != 0:
                sub_size = self.batch_size
                q -= 1
            elif r != 0:
                sub_size = r
                r = 0
            sub_x, sub_y, sub_mask = dataset_validation(sub_size)
            feed_dict = {'x':sub_x, 'y':sub_y, 'mask':sub_mask}
            sub_eval = self.model.evaluation(feed_dict)
            sub_pred = sub_eval.pop('prediction')
            evaluation = U.add_dict(evaluation, U.multiply_dict(sub_eval, float(sub_size) / v_size))
            if path is not None:
                if imgs is None:
                    imgs = U.combine_images(sub_x, sub_y, sub_pred)
                elif len(imgs) < 10:
                    imgs = np.concatenate((imgs, U.combine_images(sub_x, sub_y, sub_pred)), axis = 1)

        if imgs is not None:
            U.save_images(imgs, path)

        # print
        output_str = U.eval_to_str(evaluation)
        print('validation %s'%output_str)
        return evaluation, imgs

    def _store_prediction_3d(self, dataset_validation, path):
        evaluation_o = None
        imgs_o = []
        for i in range(dataset_validation.size()):
            v_x, v_y, v_mask = dataset_validation(1)
            feed_dict = {'x':v_x, 'y':v_y, 'mask':v_mask}
            evaluation = self.model.evaluation(feed_dict)
            pred = evaluation.pop('prediction')
            evaluation_o = U.add_dict(evaluation_o, evaluation)

            # save
            imgs = U.combine_images(v_x, v_y, pred, v_mask)
            if len(imgs_o) < 3:
                U.save_images(imgs, '%s_batch_%d'%(path, i))
                imgs_o.append(imgs)

        evaluation_o = U.div_dict(evaluation_o, dataset_validation.size())

        # print
        output_str = U.eval_to_str(evaluation_o)
        print('validation %s'%output_str)

        return evaluation_o, imgs_o

    def write_summary(self, summary, writer):
        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            for key in summary:
                value = summary.get(key)
                if value.size <= 1:
                    tf.contrib.summary.scalar(key, value)
                else:
                    for i, v in enumerate(value):
                        tf.contrib.summary.scalar('%s/class_%s'%(key, i), v)

    def write_image_summary(self, imgs, writer):
        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            if type(imgs[0]) == list:
                for i, img in enumerate(imgs):
                    self.write_one_image_summary(img, 'output/sub_%s'%i)
            else:
                self.write_one_image_summary(imgs, 'output')

    def write_one_image_summary(self, imgs, path):
        for i in range(len(imgs)):
                img = imgs[i]
                img = img.reshape((1, img.shape[0], img.shape[1], 1))
                if i < len(imgs) - 1:
                    tf.contrib.summary.image('%s/class_%d'%(path, i), img)
                else:
                    tf.contrib.summary.image('%s/argmax'%path, img)
