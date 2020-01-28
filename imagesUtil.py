import cv2
import numpy as np

#To calculate the Euclidean Distance between two vectors
def calculateDistance(i1, i2):
        #return np.sum((i1-i2)**2)
        dist = np.linalg.norm(i1-i2)
        return dist

#This function returns a list of 9 rotated images
def getRotatedImages(image, degrees=[0, 45, 90, 135, 180, 225, 270, 315, 360]):

    rotatedImages = []

    for degree in degrees:

        num_rows, num_cols = image.shape[:2]

        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -degree, 1)
        img_rotation = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))

        rotatedImages.append(img_rotation)

    return rotatedImages

#This function returns a list of 9 translated images
def getTranslatedImages(image):

    resizedImages = []

    originalImage = image

    originalDim = originalImage.shape
    originalx = originalDim[0]
    float(originalx)
    originaly = originalDim[1]
    float(originaly)

    verticalTranslationWindows = []

    stepx = originalx*0.2/8
    stepy = originaly*0.2/4

    x0 = 0
    x1 = originalx - (stepx*8)
    y0 = stepy*4
    y1 = originaly - (stepy*4)

    startingWindow = ( x0, x1, y0, y1)
    verticalTranslationWindows.append(startingWindow)

    x0Movement = x0
    x1Movement = x1
    y0Movement = y0
    y1Movement = y1

    while len(verticalTranslationWindows) <9:
            x0Movement += stepx
            x1Movement += stepx
            y0Movement = y0
            y1Movement = y1

            a = (x0Movement,x1Movement,y0Movement,y1Movement)
            verticalTranslationWindows.append(a)


    for x in verticalTranslationWindows:
        x0,x1,y0,y1 = x
        cropped_image = image[ int(x0):int(x1), int(y0):int(y1)]

        img = cropped_image

        dim = originalDim[:2]
        dim1 = (dim[1], dim[0])

        resized = cv2.resize(img, dim1, interpolation = cv2.INTER_AREA)
        resizedImages.append(resized)

    return resizedImages

#This function returns a list of 9 scaled images
def getScaledImages(image):
    resizedImages = []

    originalImage = image

    originalDim = originalImage.shape
    originalx = originalDim[0]
    float(originalx)
    originaly = originalDim[1]
    float(originaly)

    ZoomingInWindows = []

    stepx = originalx*0.2/5
    stepy = originaly*0.2/5

    x0 = 0
    x1 = originalx
    y0 = 0
    y1 = originaly

    startingWindow = ( x0, x1, y0, y1)
    ZoomingInWindows.append(startingWindow)

    x0Scaling = x0
    x1Scaling = x1
    y0Scaling = y0
    y1Scaling = y1

    while len(ZoomingInWindows) < 9:
            x0Scaling += stepx
            x1Scaling -= stepx
            y0Scaling += stepy
            y1Scaling -= stepy

            a = (x0Scaling,x1Scaling,y0Scaling,y1Scaling)
            ZoomingInWindows.append(a)

    for x in (ZoomingInWindows):
        x0,x1,y0,y1 = x
        cropped_image = image[ int(x0):int(x1), int(y0):int(y1)]

        img = cropped_image

        dim = originalDim[:2]
        dim1 = (dim[1], dim[0])

        resized = cv2.resize(img, dim1, interpolation = cv2.INTER_AREA)
        resizedImages.append(resized)

    return resizedImages

#This function can only return the visualization of the transformations on a list of original images
#The purpose is to get three images similar to the ones in Lecture 9
def getThreeTransformationsImages(theImages):

    finalRotationsImage = None
    finalTranslationsImage = None
    finalScalingImage = None

    allRotations = []
    allTranslations = []
    allScaled = []

    for i in range(3):
        if i == 0:
            allRotationsList = theImages[0:9]
        if i == 1:
            allRotationsList = theImages[27:36]
        if i == 2:
            allRotationsList = theImages[54:63]

        if i == 0:
            alltranslationsList = theImages[18:27]
        if i == 1:
            alltranslationsList = theImages[45:54]
        if i == 2:
            alltranslationsList = theImages[72:81]

        if i == 0:
            allScaledList = theImages[9:18]
        if i == 1:
            allScaledList = theImages[36:45]
        if i == 2:
            allScaledList = theImages[63:72]

        if i == 0:
            allrotatedOneImage = theImages[0]
            allTranslatedOneImage = theImages[18]
            allScaledOneImage = theImages[9]
        if i == 1:
            allrotatedOneImage = theImages[27]
            allTranslatedOneImage = theImages[45]
            allScaledOneImage = theImages[36]
        if i == 2:
            allrotatedOneImage = theImages[54]
            allTranslatedOneImage = theImages[72]
            allScaledOneImage = theImages[63]


        for rotation in allRotationsList:
            allrotatedOneImage = np.concatenate((allrotatedOneImage, rotation), axis=1)
        allRotations.append(allrotatedOneImage)

        for translation in alltranslationsList:
            allTranslatedOneImage = np.concatenate((allTranslatedOneImage, translation), axis=1)
        allTranslations.append(allTranslatedOneImage)

        for scaling in allScaledList:
            allScaledOneImage = np.concatenate((allScaledOneImage, scaling), axis=1)
        allScaled.append(allScaledOneImage)

    finalRotationsImage = allRotations[0]
    finalTranslationsImage = allTranslations[0]
    finalScalingImage = allScaled[0]

    for x in allRotations[1:]:
        finalRotationsImage = np.concatenate((finalRotationsImage, x), axis=0)

    for x in allTranslations[1:]:
        finalTranslationsImage = np.concatenate((finalTranslationsImage, x), axis=0)

    for x in allScaled[1:]:
        finalScalingImage = np.concatenate((finalScalingImage, x), axis=0)

    return finalRotationsImage, finalTranslationsImage, finalScalingImage
