from preprocess import *
import os,cv2,time
import time,threading
import numpy as np

def task3_generator(root_path,phase,batch_size):

    xy_size = 128
    z_size = 64
  
    class_num = 3
    data_path = root_path+'Task03_Liver/spacing2.5/'+phase+'/imagesTr/'
    mask_path = root_path+'Task03_Liver/spacing2.5/'+phase+'/labelsTr/'
  
    datas = np.zeros([batch_size,xy_size,xy_size,z_size,1])
    masks = np.zeros([batch_size,xy_size,xy_size,z_size,1])
   
    filenames = os.listdir(data_path)
    for k in range(random.randint(0,20)):
        random.shuffle(filenames)
    length = len(filenames) 
    cur = 0
    i = 0

    while 1:
        img,origin,spacing,direction = load_image(data_path+filenames[cur])#x,y,z,c 
        img = window_level_normalization(img,25,450)

        mask,_,_,_ = load_image(mask_path+filenames[cur])  
        shape_x,shape_y,shape_z = img.shape

           
        if phase == 'train':
            scale = random.uniform(0.8,1.2)
            old_shape = np.asarray([shape_x, shape_y, shape_z]) 
            new_shape = (old_shape*scale).astype('int')
            new_spacing = spacing/scale
      
            img = zoom(img,new_shape, new_spacing, spacing, origin, direction)
            mask = zoom(mask,new_shape, new_spacing, spacing, origin, direction)
            mask = np.round(mask)
        

        shape_x,shape_y,shape_z = img.shape 

        if shape_x <xy_size or shape_y<xy_size or shape_z<z_size:
            cur += 1
            if cur >= length:
                for k in range(random.randint(0,20)):
                    random.shuffle(filenames)
                cur = 0
            continue 
 
        #compute mask center
        mask_x,mask_y,mask_z = np.where(mask)
        mask_center_x = int(mask_x.mean())
        mask_center_y = int(mask_y.mean())
        mask_center_z = int(mask_z.mean())

        
        if phase == 'train':   
            mask_move_x = (mask_x.max()-mask_x.min())//1.5
            mask_move_y = (mask_y.max()-mask_y.min())//1.5
            mask_move_z = (mask_z.max()-mask_z.min())//1.5
        else:
            mask_move_x = (mask_x.max()-mask_x.min())//2
            mask_move_y = (mask_y.max()-mask_y.min())//2
            mask_move_z = (mask_z.max()-mask_z.min())//2

        if phase == 'train':
            sample = 2
        if phase == 'val':
            sample = 4
        s = 0 
        while s < sample:
         
            mask_center_x = random.randint( min(max(xy_size//2,mask_center_x-mask_move_x),shape_x-xy_size//2), max(min(mask_center_x+mask_move_x,shape_x-xy_size//2),xy_size//2) )
            mask_center_y = random.randint( min(max(xy_size//2,mask_center_y-mask_move_y),shape_y-xy_size//2), max(min(mask_center_y+mask_move_y,shape_y-xy_size//2),xy_size//2) )
            mask_center_z = random.randint( min(max(z_size//2,mask_center_z-mask_move_z),shape_z-z_size//2) , max(min(mask_center_z+mask_move_z,shape_z-z_size//2),z_size//2) )
                  
            crop_x_min = mask_center_x - xy_size//2                    
            crop_x_max = mask_center_x + xy_size//2                    
            crop_y_min = mask_center_y - xy_size//2                    
            crop_y_max = mask_center_y + xy_size//2                    
            crop_z_min = mask_center_z - z_size//2                    
            crop_z_max = mask_center_z + z_size//2                    

            patch_img = img[crop_x_min:crop_x_max, crop_y_min:crop_y_max, crop_z_min:crop_z_max]
            patch_mask = mask[crop_x_min:crop_x_max, crop_y_min:crop_y_max, crop_z_min:crop_z_max]

           
            if patch_mask.sum()<10:
                continue 

            datas[i,:,:,:,0] = patch_img
            masks[i,:,:,:,0] = patch_mask

            s += 1
            i += 1
            if i >= batch_size:
                yield ({'input':datas}, {'output':masks})
                i = 0

        cur += 1
        if cur >= length:
            for k in range(random.randint(0,20)):
                random.shuffle(filenames)
            cur = 0

