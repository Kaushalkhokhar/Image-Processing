import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.image as mpimg 

# Read Images 
def image_read(source):
    img = cv2.imread(source)
    return img
# Output Images 
def image_show(img):
    plt.imshow(img) 

'''img = image_read('D:\Programming\Python\VS Code\Fuse___8.jpeg') 
image_show(img)

# To convert image to gray scale  
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
image_show(img_gray)

# To Define the contours
im2, contours, hierarchy = cv2.findContours(img_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
contours

# print the no of shapes found
print("No of shapes: {}".format(len(contours))) 

for cnt in contours:
    react = cv2.contourArea(cnt)
    box = cv2.boxPoints(react)
    box = np.int0(box)
    img = cv2.drawContours(img, [box], 0, (0, 255, 0), 3)

plt.figure(figsize=(15,12))
plt.imshow(img)
plt.title('Contors of image')
plt.show()'''


# Contor Exersize  
def image_operations():    
    im1 = image_read('D:\Programming\Python\VS Code\Fuse___8.jpeg') 
    im2 = image_read('D:\Programming\Python\VS Code\DSC_0902.JPG')
    image_show(im1)

    # Image shape and datatype
    print('im1 shape:',im1.shape)
    print('im2 shape:',im2.shape)
    print('im1 dtype:',im1.dtype)

    # to Extract a part of image by slicing
    im1[50:100,75:100]
    plt.imshow(im1[250:1000,1250:1750])

    # to get BGR of image
    print('BGR value:',im1[15,16])
    added_img = cv2.addWeighted(im1[500:1500,500:1500],0.2,im2[500:1500,500:1500],0.7,1) 
    plt.imshow(added_img)  

image_operations()







help(cv2.cvtColor)
