'''
Created on Nov 25, 2016

@author: uid38420
'''
def faceOnly(img,faces):
    faceImg = 0
    multiplier = 0.35
    for (x,y,w,h) in faces:  
        var= (y-multiplier*h)
        if var<0:
            var=0
        faceImg = img[var:y+h+multiplier*h,x-multiplier*h:x+w+multiplier*h]
    return faceImg