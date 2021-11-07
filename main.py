from keras.optimizers import *
from keras.callbacks import ModelCheckpoint,TensorBoard
import os,random
import numpy as np
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
import SimpleITK as sitk
from skimage.measure import label,regionprops
from scipy.ndimage.morphology import binary_fill_holes

from task3_generator import task3_generator
from network import network
from preprocess import *

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=True, save_weights_only=True,
                 mode='min', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)



class task3_network(object):
    def __init__(self):
        self.bn_learn = False
        self.model = network(bn_learn=self.bn_learn,weight_decay=0.00001)

    def dice_coef(self,y_true, y_pred): #dice accuracy
        y_pred = tf.where(tf.greater(y_pred, 0.5), tf.ones_like(y_pred), tf.zeros_like(y_pred))
        intersection = K.sum(y_true*y_pred, axis=[1,2,3,4])
        smooth = 1.
        return K.mean( (2.*intersection+smooth)/(K.sum(y_true, axis=[1,2,3,4])+K.sum(y_pred, axis=[1,2,3,4])+smooth) )

    def dice_coef_loss(self,y_true, y_pred): #dice accuracy
        intersection = K.sum(y_true*y_pred, axis=[1,2,3,4])
        smooth = 1.
        dice_coef = (2.*intersection+smooth)/(K.sum(y_true, axis=[1,2,3,4])+K.sum(y_pred, axis=[1,2,3,4])+smooth) 
        return K.mean(1.- dice_coef)     


    def focal_loss(self,y_true, y_pred):
        gamma = 2.0
        label         = y_true
        classification = y_pred
        _epsilon =  tf.convert_to_tensor(0.00001, 'float32')
        classification = tf.clip_by_value(classification, _epsilon, 1.-_epsilon)

        focal_weight = tf.where(tf.equal(label, 1), 1. - classification, classification)
        focal_weight = K.pow(focal_weight,gamma)
        loss_logist = tf.where(tf.equal(label, 1), classification, 1.-classification)
        crossentropy_loss = -K.log(loss_logist)
        focal_loss = focal_weight * crossentropy_loss

        miss_ind = tf.where(tf.greater(crossentropy_loss, -K.log(0.5)), tf.ones_like(crossentropy_loss), tf.zeros_like(crossentropy_loss))

        AB = K.sum(miss_ind, axis=[1,2,3,4])+1.

        focal_loss = K.sum(focal_loss, axis=[1,2,3,4])/K.pow(AB,0.5)

        return K.mean(focal_loss)

    def train(self):
   
        self.parallel_model = multi_gpu_model(self.model,gpus=2)   
   
        self.parallel_model.compile(optimizer=Adam(lr=0.0001), loss={'output': self.focal_loss},metrics={'output':self.dice_coef})

        train_generator = task3_generator("xxxx",'train',8)
        val_generator = task3_generator("xxxx",'val',8)

        check_point = ParallelModelCheckpoint(model=self.model,filepath='model1/model-{epoch:03d}-{val_loss:.4f}-{val_dice_coef:.4f}.hdf5', monitor='val_dice_coef', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
        tensorboard = TensorBoard(log_dir='logs/', write_graph=True)
        self.parallel_model.fit_generator(train_generator, steps_per_epoch=109, epochs=1000, verbose=1,callbacks=[check_point,tensorboard], validation_data=val_generator, validation_steps=22,workers=1)
    
 
    def predict(self):
        val_img_path = 'XXX/MSD/Task03_Liver/imagesTs/'
        val_mask_path = 'xxx/MSD/Task03_Liver/val/labelsTr/'
        filenames = os.listdir(val_img_path)
        filenames = filenames[53:]

        for name in filenames:
            print name
            if os.path.exists('result_test/'+name) or name=='liver_187.nii.gz':
                print 'skip', name
                continue
            img,origin,spacing,direction = load_image(val_img_path+name)
            img = window_level_normalization(img,25,450)

            scale = np.asarray(spacing/2.5)
            old_shape = img.shape
            new_shape = (old_shape*scale).astype('int')
            new_spacing = np.asarray([2.5,2.5,2.5])

            img = zoom(img,new_shape, new_spacing, spacing, origin, direction)
   
            img = np.pad(img,((20,20),(20,20),(20,20)),'constant')

            windows = [144,144,96]
            strides = [108,108,72]
            pad_xy = (144-108)/2
            pad_z = (96-72)/2

            x_position,y_position,z_position = calculate(img.shape,windows,strides)

            result = np.zeros_like(img)
            for x in x_position:
                for y in y_position:
                    for z in z_position:
                        patch_img = img[x:x+windows[0], y:y+windows[1], z:z+windows[2]]
                        patch_img.shape = [1,windows[0],windows[1],windows[2],1]

                        patch_result1 = self.model1.predict(patch_img,batch_size=1)
                        patch_result2 = self.model2.predict(patch_img,batch_size=1)
                        patch_result3 = self.model3.predict(patch_img,batch_size=1)
                        patch_result = (patch_result1+patch_result2+patch_result3)/3.0
                        patch_result = np.squeeze(patch_result)
                        patch_result[patch_result>0.5] = 1
                        patch_result[patch_result<=0.5] = 0
 
                        result[x+pad_xy:x+pad_xy+strides[0],y+pad_xy:y+pad_xy+strides[1],z+pad_z:z+pad_z+strides[2]] = patch_result[pad_xy:pad_xy+strides[0],pad_xy:pad_xy+strides[1],pad_z:pad_z+strides[2]]
            

            ConnectMap=label(result, connectivity= 2)

            max_label = 0
            max_area = 0
            for pos in range(1,int(ConnectMap.max())+1):
                cur_area = (ConnectMap==pos).astype('float').sum()
                if cur_area>max_area:
                    max_area = cur_area
                    max_label = pos                
            
            result = (ConnectMap==max_label)*1.0

            result = result[20:20+new_shape[0],20:20+new_shape[1],20:20+new_shape[2]]
            result = zoom(result,old_shape, spacing, new_spacing, origin, direction) 
            print 'zoom2',result.shape
            result = np.round(result)*1.0
            result = binary_fill_holes(result)*1.0

            result = result.swapaxes(0,2)
            sitk_img = sitk.GetImageFromArray(result)
            sitk_img.SetOrigin(origin)
            sitk_img.SetSpacing(spacing)
            sitk_img.SetDirection(direction)            
            sitk.WriteImage(sitk_img,'result_test/'+name)


if __name__ == '__main__':
    Liver_task = task3_network()
    Liver_task.train()
    
