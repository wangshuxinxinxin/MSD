import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
from keras.models import *
from keras.layers.normalization import BatchNormalization
from keras.layers import Input,Add,Concatenate
from keras.layers import Conv3D, UpSampling3D,Conv3DTranspose
from keras.layers import Reshape,Concatenate,Multiply
from keras.layers.pooling import MaxPooling3D, AveragePooling3D,GlobalAveragePooling3D
from keras.layers.core import Activation, Dense, Flatten
from keras import regularizers
from keras.utils import plot_model


def SE_block(x,pre_name,C):
    x1 = GlobalAveragePooling3D()(x)
    x1 = Dense(C/2,use_bias=False,name=pre_name+'W1',activation='relu')(x1)
    x1 = Dense(C,use_bias=False,name=pre_name+'W2',activation='sigmoid')(x1)
    x1 = Reshape((1,1,1,C),name=pre_name+'reshape')(x1)
    x = Multiply(name=pre_name+'Multiply')([x,x1])
    return x


def DUC_block(x,pre_name,num_output,bn_learn,weight_decay):

    x0 = BatchNormalization(trainable=bn_learn,name=pre_name + 'BN0')(x) 
    
    x1 = Conv3D(filters=num_output, kernel_size=3, strides=1,padding='same',activation='relu',name=pre_name + 'conv1',dilation_rate=1,kernel_regularizer=regularizers.l2(weight_decay))(x0)
    x1 = Conv3D(filters=num_output, kernel_size=3, strides=1,padding='same',activation='relu',name=pre_name + 'conv2',dilation_rate=2,kernel_regularizer=regularizers.l2(weight_decay))(x1)
    x1 = Conv3D(filters=num_output, kernel_size=3, strides=1,padding='same',name=pre_name + 'conv3',dilation_rate=4,kernel_regularizer=regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization(trainable=bn_learn,name=pre_name + 'BN1')(x1) 
     
    x = Add(name=pre_name+'add')([x0,x1])
    x = Activation('relu',name=pre_name + 'ReLU')(x)
    return x
    

def residual_block(x,pre_name,num_output,bn_learn,weight_decay):
  
    x1 = BatchNormalization(trainable=bn_learn,name=pre_name + 'BN1')(x) 
    x1 = Activation('relu',name=pre_name + 'ReLU1')(x1)
    x1 = Conv3D(filters=num_output, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name=pre_name + 'conv1')(x1) 

    x2 = BatchNormalization(trainable=bn_learn,name=pre_name + 'BN2')(x1) 
    x2 = Activation('relu',name=pre_name + 'ReLU2')(x2)
    x2 = Conv3D(filters=num_output, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name=pre_name + 'conv2')(x2) 
     
    x = Add(name=pre_name+'add')([x,x2])
    return x


def network(bn_learn,weight_decay,data_channel=1,nclass = 3):

    network_input = Input((None,None,None,data_channel), name='input')
    #network_input = Input((None, None, None, data_channel), name='input_')

    x0 = Conv3D(filters=64, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name='conv_block0_1')(network_input)
    #x0 = Conv3D(filters=64, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay),
     #           name='conv_block0_1_')(network_input)
    x0 = DUC_block(x0,'DUC_block0_',64,bn_learn,weight_decay)  
    x0 = SE_block(x0,'SE_block0_',64)
     

    x1 = MaxPooling3D(pool_size=(2, 2, 2),name='pool1')(x0)

    x1 = Conv3D(filters=64, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name='conv_block1_1')(x1)
    x1 = DUC_block(x1,'DUC_block1_',64, bn_learn, weight_decay)
    x1 = SE_block(x1,'SE_block1_',64)

    x2 = MaxPooling3D(pool_size=(2, 2, 2),name='pool2')(x1)

    x2 = Conv3D(filters=128, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name='conv_block2_1')(x2)
    x2 = residual_block(x2,'residual2_1_',128,bn_learn,weight_decay)
    x2 = residual_block(x2,'residual2_2_',128,bn_learn,weight_decay)
    x2 = SE_block(x2,'SE_block2_',128)

    x3 = MaxPooling3D(pool_size=(2, 2, 2),name='pool3')(x2)

    x3 = Conv3D(filters=128, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name='conv_block3_1')(x3)
    x3 = residual_block(x3,'residual3_1_',128,bn_learn,weight_decay)
    x3 = residual_block(x3,'residual3_2_',128,bn_learn,weight_decay)
    x3 = SE_block(x3,'SE_block3_',128)

    up2 = UpSampling3D(size=(2, 2, 2),name='uppool2')(x3) 
    up2 = Conv3D(filters=128, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name='conv_up2_0')(up2)
    up2 = Add(name='add2')([x2,up2])           
    up2 = Conv3D(filters=128, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name='conv_up2_1')(up2)
    up2 = residual_block(up2,'residualup2_1_',128,bn_learn,weight_decay)
    up2 = residual_block(up2,'residualup2_2_',128,bn_learn,weight_decay)
    up2 = SE_block(up2,'SE_blockup2_',128)

    up1 = UpSampling3D(size=(2, 2, 2),name='uppool1')(up2) 
    up1 = Conv3D(filters=64, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name='conv_up1_0')(up1)
    up1 = Add(name='add1')([x1,up1])           
    up1 = Conv3D(filters=64, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name='conv_up1_1')(up1)
    up1 = residual_block(up1,'residualup1_1_',64,bn_learn,weight_decay)
    up1 = residual_block(up1,'residualup1_2_',64,bn_learn,weight_decay)
    up1 = SE_block(up1,'SE_blockup1_',64)

    up0 = UpSampling3D(size=(2, 2, 2),name='uppool0')(up1) 
    up0 = Conv3D(filters=64, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name='conv_up0_0')(up0)
    up0 = Add(name='add0')([x0,up0])           
    up0 = Conv3D(filters=64, kernel_size=3, strides=1,padding='same',kernel_regularizer=regularizers.l2(weight_decay),name='conv_up0_1')(up0)
    up0 = residual_block(up0,'residualup0_1_',64,bn_learn,weight_decay)
    up0 = residual_block(up0,'residualup0_2_',64,bn_learn,weight_decay)
    up0 = SE_block(up0,'SE_blockup0_',64)

    if nclass == 1:
        main_output = Conv3D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid',
                             name='output')(up0)
    else:
        main_output = Conv3D(filters=nclass, kernel_size=1, strides=1, padding='same', activation='softmax',
                             name='output')(up0)

    model = Model(inputs = [network_input], outputs = [main_output]) 
        
    return model


if __name__ == "__main__":
    net = network(True,0.00001,3)
    #plot_model(net, to_file='model.png',show_shapes=True)
    print net.summary()

