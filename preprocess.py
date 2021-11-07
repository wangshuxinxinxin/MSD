import numpy as np
import SimpleITK as sitk
import random
from scipy.ndimage.interpolation import rotate
from skimage import exposure
import nibabel as nib
from skimage import measure

def calculate(img_shape,windows,strides):
    xs = []
    ys = []
    zs = []

    for x in range(0, img_shape[0]-windows[0], strides[0]):
        xs.append(x)
    xs.append(img_shape[0]-windows[0])

    for y in range(0,img_shape[1]-windows[1],strides[1]):
        ys.append(y)
    ys.append(img_shape[1] - windows[1])

    for z in range(0,img_shape[2]-windows[2],strides[2]):
        zs.append(z)
    zs.append(img_shape[2] - windows[2])
    return [xs, ys, zs]



def window_level_normalization(img,level,window):
    min_HU = level - window/2
    max_HU = level + window/2
    img[img>max_HU] = max_HU
    img[img<min_HU] = min_HU
    img = 1.*(img-min_HU)/(max_HU-min_HU)
    return img
   

def mask_split(mask,class_num):
    masks = []
    for c in range(class_num):
        masks.append( np.asarray(mask==c,dtype='int'))
    return masks

def mask_to_onehot(mask,nclass):
    x, y, z = mask.shape
    mask_onehot = np.zeros([x, y, z, nclass])

    mask_onehot[:, :, :, 0] = np.ones([x, y, z])
    for i in range(1, nclass):
        seg_one = mask == i
        mask_onehot[:, :, :, i] = seg_one[0:x, 0:y, 0:z]
        mask_onehot[:, :, :, 0] = mask_onehot[:, :, :, 0] - mask_onehot[:, :, :, i]

    mask_onehot = mask_onehot.astype(np.float32)

    return mask_onehot

def zoom(img, new_shape, new_spacing, old_spacing, origin, direction):

    img = img.swapaxes(0,2) #x,y,z -> z,y,x
    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetOrigin(origin)
    sitk_img.SetSpacing(old_spacing)
    sitk_img.SetDirection(direction)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_shape)
    newimage = resample.Execute(sitk_img)
    return sitk.GetArrayFromImage(newimage).swapaxes(0,2)# get img z,y,x -> x,y,z

def load_image(fileName): 
    itk_img = sitk.ReadImage(fileName)
    img = sitk.GetArrayFromImage(itk_img).swapaxes(0,2) # get img z,y,x -> x,y,z
    origin = itk_img.GetOrigin()
    spacing = itk_img.GetSpacing()
    direction = itk_img.GetDirection()
    return img.astype('float32'), np.asarray(origin), np.asarray(spacing), direction

def load_nii_image(fileName):
    testImage = nib.load(fileName)
    img = testImage.get_fdata()
    return img

#def random_rotate(img):
#    anagle = [0,90,180,270]
#    random.shuffle(anagle)
#    return rotate(img,angle=anagle[0],axes=(0,1))

def flip(img,x,y,z):
    if x:
        img = img[::-1,:,:]

    if y:
        img = img[:,::-1,:]

    if z:
        img = img[:,:,::-1]

    return img


def clahe(image,need_clahe):
    image = image.astype(np.float32)
    image -= np.min(image)
    image /= np.max(image)

    shape_x,shape_y,shape_z = image.shape
    ret = np.zeros(image.shape)
    
    for slice_num in need_clahe:
        if image[:, :, slice_num].mean()!=0:
            ret[:, :, slice_num] = exposure.equalize_adapthist(image[:, :, slice_num], clip_limit=0.03)
    
    return ret

def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""
    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
#        print("min", np.min(ret), "max", np.max(ret), "mean", np.mean(ret), "std", np.std(ret))
    else:
        ret = image * 0.
    return ret


def CustomCenterCrop(image, mask, size):
    size_x, size_y, size_z = size
    x, y, z = image.shape
    if x < size_x or y < size_y or z < size_z:
        raise ValueError
    # center_x = random.randint(x // 2 - 10, x // 2 + 10)
    # center_y = random.randint(y // 2 - 10, y // 2 + 10)
    # if center_x + size_x // 2 > x:
    #     center_x = x - (center_x + size_x // 2)
    # if center_y + size_y // 2 > y:
    #     center_y = y - (center_y + size_y // 2)

    result_mask = np.copy(mask)
    ConnectMap = measure.label(result_mask, connectivity=2)
    max_label = 0
    max_area = 0
    for pos in range(1, int(ConnectMap.max()) + 1):
        cur_area = (ConnectMap == pos).astype('float').sum()
        if cur_area > max_area:
            max_area = cur_area
            max_label = pos
    result = (ConnectMap == max_label) * 1.0
    mask_x, mask_y, mask_z = np.where(result)

    x1 = x // 2 - size_x // 2
    y1 = y // 2 - size_y // 2
    if mask_z.min() + size_z > z:
        z1 = mask_z.min() - (mask_z.min() + size_z - z)
    else:
        z1 = mask_z.min()

    # x1 = max(center_x - size_x // 2, 0)
    # y1 = max(center_y - size_y // 2, 0)
    # z1 = max(center_z - size_z // 2, 0)

    image = image[x1: x1 + size_x, y1: y1 + size_y, z1: z1 + size_z]
    label = mask[x1: x1 + size_x, y1: y1 + size_y, z1: z1 + size_z]

    return np.array(image), np.array(label)

def CustomSpecialCenterCrop(image, label, size, class_num):
    target_x, target_y, target_z = size
    x, y, z= image.shape
    if x < target_x or y < target_y or z < target_z:
        raise ValueError

    while 1:

        mask0, mask1, mask2 = mask_split(label, class_num)

        #calculate mask certer
        mask1 = mask1 + mask2
        mask_x, mask_y, mask_z = np.where(mask1)
        mask_center_x = int(mask_x.mean())
        mask_center_y = int(mask_y.mean())
        mask_center_z = int(mask_z.mean())

        crop_x_min = mask_center_x - target_x // 2
        crop_x_max = mask_center_x + target_x // 2
        crop_y_min = mask_center_y - target_y // 2
        crop_y_max = mask_center_y + target_y // 2
        crop_z_min = mask_center_z - target_z // 2
        crop_z_max = mask_center_z + target_z // 2

        patch_img = image[crop_x_min:crop_x_max, crop_y_min:crop_y_max, crop_z_min:crop_z_max]
        patch_mask = label[crop_x_min:crop_x_max, crop_y_min:crop_y_max, crop_z_min:crop_z_max]

        if patch_mask.sum() < 100:
            continue
        else:
            return np.array(patch_img), np.array(patch_mask)

def CustomRandomCrop(image, label, size):
    size_x, size_y, size_z = size
    x, y, z = image.shape
    if x < size_x or y < size_y or z < size_z:
        raise ValueError

    x1 = random.randint(0, x - size_x)
    y1 = random.randint(0, y - size_y)
    z1 = random.randint(0, z - size_z)

    image = image[x1: x1 + size_x, y1: y1 + size_y, z1: z1 + size_z]
    label = label[x1: x1 + size_x, y1: y1 + size_y, z1: z1 + size_z]

    return np.array(image), np.array(label)

def CustomSpecialRandomCrop(image, label, size, class_num):
    target_x, target_y, target_z = size
    x, y, z= image.shape
    if x < target_x or y < target_y or z < target_z:
        raise ValueError

    while 1:

        mask0, mask1, mask2 = mask_split(label, class_num)
        shape_x, shape_y, shape_z = image.shape

        #calculate mask certer
        mask1 = mask1 + mask2
        mask_x, mask_y, mask_z = np.where(mask1)
        mask_center_x = int(mask_x.mean())
        mask_center_y = int(mask_y.mean())
        mask_center_z = int(mask_z.mean())

        mask_move_x = (mask_x.max() - mask_x.min()) // 2
        mask_move_y = (mask_y.max() - mask_y.min()) // 2
        mask_move_z = (mask_z.max() - mask_z.min()) // 2

        mask_center_x = random.randint(min(max(target_x // 2, mask_center_x - mask_move_x), shape_x - target_x // 2),
                                       max(min(mask_center_x + mask_move_x, shape_x - target_x // 2), target_x // 2))
        mask_center_y = random.randint(min(max(target_y // 2, mask_center_y - mask_move_y), shape_y - target_y // 2),
                                       max(min(mask_center_y + mask_move_y, shape_y - target_y // 2), target_y // 2))
        mask_center_z = random.randint(min(max(target_z // 2, mask_center_z - mask_move_z), shape_z - target_z // 2),
                                       max(min(mask_center_z + mask_move_z, shape_z - target_z // 2), target_z // 2))

        crop_x_min = mask_center_x - target_x // 2
        crop_x_max = mask_center_x + target_x // 2
        crop_y_min = mask_center_y - target_y // 2
        crop_y_max = mask_center_y + target_y // 2
        crop_z_min = mask_center_z - target_z // 2
        crop_z_max = mask_center_z + target_z // 2

        patch_img = image[crop_x_min:crop_x_max, crop_y_min:crop_y_max, crop_z_min:crop_z_max]
        patch_mask = label[crop_x_min:crop_x_max, crop_y_min:crop_y_max, crop_z_min:crop_z_max]

        if patch_mask.sum() < 10:
            continue
        else:
            return np.array(patch_img), np.array(patch_mask)


# xs, ys, zs = calculate([512,512,150],[192,192,48],[128,128,34])
# print(xs)
# # xs_, ys_, zs_ = calculate([240,240,150],[100,100,48],[48,48,34])
# # print(xs_)


