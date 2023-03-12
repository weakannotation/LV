import cv2
import numpy as np
import pydicom, cv2, re, math, shutil
import os, fnmatch, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Lambda
from keras.layers import Input, average
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import ZeroPadding2D, Cropping2D
from keras.losses import categorical_crossentropy as CCE
import tensorflow as tf
from fcn_model import dice_coef, jaccard_coef, dice_coef_loss
#from itertools import izip
from fcn_model import fcn_model, custom_loss
from helpers import center_crop, lr_poly_decay, get_SAX_SERIES
seed = 1234
np.random.seed(seed)
from weak_annotator import annotation, fill_circle, draw_circle

# image_file = "C:\\Users\\behnam\\py\\Capturelv.png"
# img = cv2.imread(image_file)
 
# image_file = "C:\\Users\\behnam\\py\\Capturelv.png"
# image = cv2.imread(image_file)
# print(image.shape)


SUNNYBROOK_ROOT_PATH = "C:\\Users\\r_beh\\data"
VAL_CONTOUR_PATH = "C:\\Users\\r_beh\\data\\valGT"
VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_validation')
ONLINE_CONTOUR_PATH = "C:\\Users\\r_beh\\data\\online"
ONLINE_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_online')
seed = 1234
np.random.seed(seed)

TRAIN_CONTOUR_PATH = "C:\\Users\\r_beh\\OneDrive\\Desktop\\backup\\backup"
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_training')
TEST_CONTOUR_PATH = "C:\\Users\\r_beh\\OneDrive\\Desktop\\backup\\test"
TEST_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_test')   
                        


class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'\\([^\\]*)\\contours-manual\\IRCCI-expert\\IM-0001-(\d{4})-.*', ctr_path)
        self.case = match.group(1)
        self.img_no = int(match.group(2))
        self.slice_no =  math.floor(self.img_no/20) if self.img_no%20 !=0 else math.floor(self.img_no/20)-1
        self.ED_flag = True if ((self.img_no%20) < 10 and (self.img_no % 20) !=0) else False
        self.is_weak = 0

    
    def __str__(self):
        return 'Contour for case %s, image %d' % (self.case, self.img_no)
    
    __repr__ = __str__
def read_contour(contour, data_path):
    filename = 'IM-0001-%04d.dcm' % ( contour.img_no)
    full_path = os.path.join(data_path, contour.case,'DICOM', filename)
    f = pydicom.dcmread(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8')

    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    
    return img, mask

def map_all_contours(contour_path, contour_type, shuffle=True):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files,
                        'IM-0001-*-'+contour_type+'contour-manual.txt')]
    #for dirpath, dirnames, files in os.walk(contour_path):
    #    print(files)
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
        
    print('Number of examples: {:d}'.format(len(contours)))
    contours = map(Contour, contours)
    
    return contours

def export_all_contours(contours, data_path, crop_size):
    print('\nProcessing {:d} images and labels ...\n'.format(len(list(contours))))
    print(len(contours))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask

    return images, masks


if __name__== '__main__':
   # if len(sys.argv) < 3:
   #     sys.exit('Usage: python %s <i/o> <gpu_id>' % sys.argv[0])
    contour_type = sys.argv[1]
    print(contour_type)
    #training_dataset= sys.argv[2]
    #os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    crop_size = 100
    print("HIII")

    print('Mapping ground truth '+contour_type+' contours to images in train...')
    a = list(map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True))

    print('Done mapping training set')

    split = int(0*len(a))
    train_ctrs=a[split:]
    #dev_ctrs = b[0:split]
    print(len(a))
    print("before")

    print(len(train_ctrs))
    #print(train_ctrs[:])
    test_ctrs = list(map_all_contours(TEST_CONTOUR_PATH, contour_type = 'i', shuffle=False))
    
    print('\nBuilding Train dataset ...')
    img_train, mask_train = export_all_contours(train_ctrs,
                                                TRAIN_IMG_PATH,
                                                crop_size=crop_size)
    print('\nBuilding Dev dataset ...')
    img_dev, mask_dev = export_all_contours(test_ctrs,
                                            TEST_IMG_PATH,
                                            crop_size=crop_size)
                                            
    random_indices_test = [1, 122, 46, 0, 45, 28, 94, 52, 60, 85, 50, 123, 38, 6, 59, 75, 53, 20, 118, 63, 7, 35, 121, 30, 101, 51, 40, 105, 79, 33, 84, 29, 76, 90, 120, 109, 112, 73, 70, 5, 18, 102, 39, 83, 47, 80, 36, 21, 68, 92]
            
    mask_val = np.array([mask_dev[i] for i in random_indices_test])

    
    img_val = np.array([img_dev[i] for i in random_indices_test])
    
    masks_weak = []
    imgs_weak = []
    masks_ignore = []
    for idx, imgs in enumerate(img_train):
        print(np.array(img_train).shape)
        imgs_m = cv2.normalize(imgs, None, 0, 255, cv2.NORM_MINMAX)
        imgs_m = imgs_m.astype('uint8')
     

        # rgb_img= np.zeros((256,256, 3), dtype=np.uint8)
        rgb_img = np.stack((imgs_m,) * 3, axis=-1)
        cv2.setMouseCallback("Image", draw_circle)
        radiuses, centers, next_flag = annotation(rgb_img)
        if (next_flag == False):

            center1 = centers[0]
            center2 = centers[1]
            radius1 = radiuses [0]
            radius2 = radiuses [1]
            # create weak_masks based on the radiuses and centers
            mask_weak = np.zeros_like(imgs, dtype='uint8')
            ignore_matrix = np.zeros_like(imgs, dtype='uint8')
            y, x = np.ogrid[:crop_size, :crop_size]        
            dist_from_center1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2)  
            dist_from_center2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2)
        
            mask_weak[dist_from_center1 <= radius1] = 1
            mask_weak[(dist_from_center1 > radius1) & (dist_from_center2 < radius2)] = 125
            ignore_matrix [mask_weak == 125] = 1

            if mask_weak.ndim < 3:
                mask_weak = mask_weak[..., np.newaxis]        
            # mask_weak = center_crop(mask, crop_size=crop_size)
            masks_weak.append(mask_weak)
            masks_ignore.append(ignore_matrix)
            imgs_weak.append(imgs)
            # cv2.imshow('mask', mask_weak)
         
    imgs_weak = np.array(imgs_weak)    
    masks_weak = np.array(masks_weak)    
    masks_ignore = np.array(masks_ignore)
    print((imgs_weak).shape)
    print((masks_weak).shape)
    print(masks_ignore.shape)
    input_shape = (crop_size, crop_size, 1)
    num_classes = 2
    # #weights = 'C:\\Users\\r_beh\\cardiac-segmentation-master\\cardiac-segmentation-master\\model_logs_backupl\\sunnybrook_i_epoch_40.h5'
    model = fcn_model(input_shape, num_classes, weights=None)
    
    # print(model.summary())
    # #model = unet(input_size = input_shape, pretrained_weights=None)    
    kwargs = dict(
        rotation_range=180,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    epochs = 40
    mini_batch_size = 1

    # image_generator = image_datagen.flow(imgs_weak, shuffle=False,
                                    # batch_size=mini_batch_size, seed=seed)
    # mask_generator = mask_datagen.flow(masks_weak, shuffle=False,
                                    # batch_size=mini_batch_size, seed=seed)
    # train_generator = zip(image_generator, mask_generator)
    ignore_matrix = np.zeros_like(imgs, dtype='uint8')
    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)          
    
    model.compile(optimizer=sgd, loss=custom_loss(ignore_matrix),
          metrics=['accuracy', dice_coef, jaccard_coef], run_eagerly = True)     
    max_iter = (len(masks_weak) / mini_batch_size) * epochs
    curr_iter = 0
    base_lr = K.eval(model.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)
    for e in range(epochs):
        print('\nMain Epoch {:d}\n'.format(e+1))
        print('\nLearning rate: {:6f}\n'.format(lrate))
        train_result = []
        # ignore_matrix = np.zeros_like(imgs, dtype='uint8')
        # model.compile(optimizer=sgd, loss=custom_loss(ignore_matrix),
              # metrics=['accuracy', dice_coef, jaccard_coef], run_eagerly = True)        
        for iteration in range(int(len(imgs_weak)/mini_batch_size)):
       
            # img, mask = next(train_generator)
            img = np.expand_dims (imgs_weak [iteration], axis = 0)
            # print(np.array(img).shape)
            mask = np.expand_dims (masks_weak [iteration], axis = 0)
            ignore_matrix = np.expand_dims(masks_ignore[iteration], axis = 0)
            model.compile(optimizer=sgd, loss=custom_loss(ignore_matrix),
                      metrics=['accuracy', dice_coef, jaccard_coef], run_eagerly = True)              
            # sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)          
            # model.compile(optimizer=sgd, loss=custom_loss(ignore_matrix),
                  # metrics=['accuracy', dice_coef, jaccard_coef], run_eagerly = True)            
            res = model.train_on_batch(img, mask)
            # print(res)
            
            curr_iter += 1
            lrate = lr_poly_decay(model, base_lr, curr_iter,
                                  max_iter, power=0.5)
            train_result.append(res)
        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)
        print(model.metrics_names, train_result)
        print('\nEvaluating dev set ...')
        model.compile(optimizer=sgd, loss=dice_coef_loss,
                   metrics=['accuracy', dice_coef, jaccard_coef], run_eagerly = True)             
        result = model.evaluate(img_val, mask_val, batch_size=32)
        result = np.round(result, decimals=10)
        print(model.metrics_names, result)
        save_file = '_'.join(['sunnybrook', contour_type,
                              'epoch', str(e+1)]) + '.h5'
        if not os.path.exists('weights'):
            os.makedirs('weights')
        save_path = os.path.join('weights', save_file)
        model.save_weights(save_path)
