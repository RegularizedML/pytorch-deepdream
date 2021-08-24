import os
import copy
from keras.preprocessing.image import load_img
from imgaug import augmenters as iaa
from tensorflow.keras.utils import Sequence
from scipy import ndimage as ndi 
from PIL import ImageOps
from itertools import cycle, islice
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms
import torch

class Dataset(Sequence):
      
  def __init__(self,path,to_fit=True,AE=True, is_val=False,input_shape=252, Eff=False):
    
    self.is_val=is_val
    self.idxList=[]    
    self.input_shape=input_shape
    self.images,self.Class= self.ImagesLoadFromPath(path,self.input_shape)
    self.to_fit=to_fit
    self.batch_size=1
    self.AE=AE
    self.numImages= self.images.shape[0]
    if self.is_val:
      self.idxList=[i for i in range(self.numImages-300,self.numImages)]
    else:
      self.idxList=[i for i in range(0,self.numImages-300)]
    self.Eff = Eff
    self.maxAray=[]
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

  def __getitem__(self, index):


    start=index*self.batch_size
    ending= index*self.batch_size+self.batch_size
    tempList=self.idxList[start:ending]
    image=[self.images[i] for i in tempList]
    Class=[self.Class[i] for i in tempList]
    image= image[0]
    #image = image.reshape((3,self.input_shape,self.input_shape))
    Class = Class[0]

    if self.is_val:
      #image_seg=  self.Segmentation(image)
      #imageDenom= np.array(image_seg)
      imageDenom= np.array(image)
      image=self.Norming(imageDenom)
    else:
      #image_seg= self.Segmentation(image)
      #imageDenom=  self.DataAugemntation(image_seg)
      #imageDenom= np.array(image)
      image=self.Norming(image)
      
      
    if self.to_fit:
      if self.AE:
        return image, {'Dec':image,}
      else:
        #print(np.asarray(Class).shape)
        return image, np.asarray(Class)
    else:
      return image
    
  #def DataAugemntation(self, images): #input should be a list of numpy arrays (list of images)
  #  Auge= iaa.RandAugment(n=(1,1),m=(10))
  #  Auge= iaa.RandAugment(n=(1,1),m=(10))
  #  out=Auge(images=images)
  #  return np.array(out)


  #def on_epoch_end(self):
    
   # np.random.seed(20)
    ## #Shuffle list magic goes here?
    #print("shuffle done!")

  def ImagesLoadFromPath(self,path,desired_size=500):
    ImagesArray=[]
    Classes= []
    folderss=os.listdir(path)
    if '.ipynb_checkpoints' in folderss: folderss.remove(".ipynb_checkpoints")
    if 'LICENSE.txt' in folderss: folderss.remove("LICENSE.txt")
    folderss.sort()
    NumberOfClasses= len(folderss)
    dummyVector=[0 for i in range(NumberOfClasses)]
    
    if not self.is_val:
      min_img= 0
    else: 
      min_img= 0
    counterIdx=0
    for i,folder in enumerate(folderss):
      
      arrayidx=[]
      dummyVectorUp=copy.deepcopy(dummyVector)
      dummyVectorUp[i]=1
      dummyVectorUp=np.asarray(dummyVectorUp)
      folderImage= os.path.join(path,folder)
      print(folder)
      for imageName in os.listdir(folderImage):
        imagePath= os.path.join(folderImage,imageName)
        im=load_img(imagePath)
        if im.size == (desired_size,desired_size):
          ImagesArray.append(np.asarray(im))
        else: 
          new_im=self.resize_with_padding(im,desired_size)
          old_size = im.size
          ImagesArray.append(np.asarray(new_im))
        Classes.append(dummyVectorUp)

        arrayidx.append(counterIdx)
        counterIdx+=1
      if len(arrayidx)<min_img:
        output = list(islice(cycle(arrayidx), min_img))
      else:
        output=arrayidx
      self.idxList.extend(output)
    return np.array(ImagesArray), np.array(Classes)

  def __len__(self):

    if self.numImages % self.batch_size:
      return int(len(self.idxList) / self.batch_size) 
    else:
      return int(len(self.idxList)  / self.batch_size)

  def resize_with_padding(self,img, expected_size):
      img.thumbnail((expected_size, expected_size))
      # print(img.size)
      delta_width = expected_size - img.size[0]
      delta_height = expected_size - img.size[1]
      pad_width = delta_width // 2
      pad_height = delta_height // 2
      padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
      return ImageOps.expand(img, padding)

  def Norming(self, img): 
    #normalized = img.astype('float64')/255.0
    if self.Eff:
      
    	#normalized = img.astype('float64')
      normalized = self.transform(img.astype('float64')).to(self.device)
    else:
      normalized = img.astype('float64')/255.0
    #for i in range(len(img)):
     #   normalized[i,:,:,:] = img[i,:,:,:].astype('float64')/255.0
    #self.maxAray.append(maxx)
    return normalized
