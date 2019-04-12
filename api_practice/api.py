import cv2
import json
import numpy as np
import dlib
import os
import api_practice.utils as utils
import api_practice.image_utils as image_utils
import random
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view

def adjustBrightness(image, percentage, scale):
    ycbImage = image_utils.loadImageYcb(image)
    beta = 75 * percentage / 100 * scale
    Ychannel, Cr, Cb = cv2.split(ycbImage)
    Ychannel = np.clip(Ychannel + beta, 0, 255)
    ycbImage = np.uint8(cv2.merge([Ychannel, Cr, Cb]))
    imbright = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)
    # combined = np.hstack([image, imbright])
    return imbright


@api_view(['POST'])
def cooler(request):

    data = utils.parseRequest(request)
    url = data['image_url']
    # percentage = data['percentage']

    original = utils.url_to_image(url)
    image = np.copy(original)

    # Pivot points for X-Coordinates
    originalValue = np.array([0, 50, 100, 150, 200, 255])
    # Changed points on Y-axis for each channel
    # bCurve = np.array([0, 80 + 35, 150 + 35, 190 + 35, 220 + 35, 255])
    # rCurve = np.array([0, 20 - 20,  40 - 20,  75 - 20, 150 + 20, 255])
    bCurve = np.array([0, 80, 150, 190, 220, 255])
    rCurve = np.array([0, 20,  40,  75, 150, 255])

    # Create a LookUp Table
    fullRange = np.arange(0, 256)
    rLUT = np.interp(fullRange, originalValue, rCurve)
    bLUT = np.interp(fullRange, originalValue, bCurve)

    # print(original)
    # print(image)
    # print(image.shape)
    bChannel = image[:, :, 0]
    bChannel = cv2.LUT(bChannel, bLUT)
    image[:, :, 0] = bChannel

    # Get the red channel and apply the mapping
    rChannel = image[:, :, 2]
    rChannel = cv2.LUT(rChannel, rLUT)
    image[:, :, 2] = rChannel

    url = utils.image_to_url("results/cooler.jpg", image)
    # cv2.imwrite("results/cooler_%s.jpg" % ('result'), image)
    # return Response('', status=status.HTTP_200_OK)
    return Response({"image_url": url})


@api_view(['POST'])
def warmer(request):
    data = utils.parseRequest(request)
    url = data['image_url']
    # percentage = data['percentage']

    original = utils.url_to_image(url)
    image = np.copy(original)

   # Pivot points for X-Coordinates
    originalValue = np.array([0, 50, 100, 150, 200, 255])
    # Changed points on Y-axis for each channel
    # rCurve = np.array([0, 80 + 35, 150 + 35, 190 + 35, 220 + 35, 255])
    # bCurve = np.array([0, 20 ,  40 ,  75 , 150, 255])

    rCurve = np.array([0, 80, 150, 190, 220, 255])
    bCurve = np.array([0, 20,  40,  75, 150, 255])

    # Create a LookUp Table
    fullRange = np.arange(0, 256)
    rLUT = np.interp(fullRange, originalValue, rCurve)
    bLUT = np.interp(fullRange, originalValue, bCurve)

    bChannel = image[:, :, 0]
    bChannel = cv2.LUT(bChannel, bLUT)
    image[:, :, 0] = bChannel

    # Get the red channel and apply the mapping
    rChannel = image[:, :, 2]
    rChannel = cv2.LUT(rChannel, rLUT)
    image[:, :, 2] = rChannel

    url = utils.image_to_url("results/warmer.jpg", image)
    # cv2.imwrite("results/warmer_%s.jpg" % ('result'), image)
    # return Response('', status=status.HTTP_200_OK)
    return Response({"image_url": url})


@api_view(['POST'])
def brighten(request):
    data = utils.parseRequest(request)
    url = data['image_url']
    percentage = data['percentage']
    # name = url.split('.')[0]
    # image = cv2.imread(url)
    image = utils.url_to_image(url)

    imbright = adjustBrightness(image, percentage, 1)
    url = utils.image_to_url("results/bright_%s_%2.2f%%.jpg" %
                             ('result', percentage), imbright)
    return Response({"image_url": url})


@api_view(['POST'])
def darken(request):
    data = utils.parseRequest(request)
    url = data['image_url']
    percentage = data['percentage']
    # name = url.split('.')[0]
    image = utils.url_to_image(url)
    imdark = adjustBrightness(image, percentage, -1)
    url = utils.image_to_url("results/dark%s_%2.2f%%.jpg" %
                             ('result', percentage), imdark)
    return Response({"image_url": url})


@api_view(['POST'])
def oldify(request):
    data = utils.parseRequest(request)
    url = data['image_url']
    # percentage = data['percentage']
    src_path = 'old_film_filter'
    src = cv2.imread('src/' + src_path + '.jpg')
    dst = utils.url_to_image(url)

    output = np.copy(dst)
    srcLab = np.float32(cv2.cvtColor(src, cv2.COLOR_BGR2LAB))

    dstLab = np.float32(cv2.cvtColor(dst, cv2.COLOR_BGR2LAB))
    outputLab = np.float32(cv2.cvtColor(output, cv2.COLOR_BGR2LAB))

    # Split the Lab images into their channels
    srcL, srcA, srcB = cv2.split(srcLab)
    dstL, dstA, dstB = cv2.split(dstLab)
    outL, outA, outB = cv2.split(outputLab)

    outL = dstL - dstL.mean()
    outA = dstA - dstA.mean()
    outB = dstB - dstB.mean()

    # scale the standard deviation of the destination image
    outL *= srcL.std() / dstL.std()
    outA *= srcA.std() / dstA.std()
    outB *= srcB.std() / dstB.std()

    # Add the mean of the source image to get the color
    outL = outL + srcL.mean()
    outA = outA + srcA.mean()
    outB = outB + srcB.mean()

    # Ensure that the image is in the range
    # as all operations have been done using float
    outL = np.clip(outL, 0, 255)
    outA = np.clip(outA, 0, 255)
    outB = np.clip(outB, 0, 255)

    # Get back the output image
    outputLab = cv2.merge([outL, outA, outB])
    outputLab = np.uint8(outputLab)

    output = cv2.cvtColor(outputLab, cv2.COLOR_LAB2BGR)

    output = output + 0.8 * output.std() * np.random.random(output.shape)
    # cv2.line(output, (450, 0), (450, 345), (0,0,0), thickness = 1, lineType=cv2.LINE_AA)
    cv2.imwrite("results/oldify_%d.jpg" % (random.randint(0, 10000)), output)
    return Response('', status=status.HTTP_200_OK)


@api_view(['POST'])
def ig_filter2(request):
    data = utils.parseRequest(request)
    url = data['image_url']
    # percentage = data['percentage']

    original = utils.url_to_image(url)
    image = np.copy(original)

    originalValue = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])

    # originalValue = np.array([0, 63, 126, 189, 255]);

    bCurve = np.array([0, 26, 62, 96, 104, 128, 153, 189, 219, 255])
    # bCurve = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # bCurve = np.array([255, 255, 255, 255, 255, 255, 255, 255, 255, 255])
    # bCurve = np.array([128, 128, 128, 128, 128, 128, 128, 128, 128, 128])
    # bCurve = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])
    # bCurve = np.array([0, 57, 128, 176, 255])

    # gCurve = np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255])
    gCurve = np.array([0, 17, 57, 65, 75, 102, 146, 172, 232, 255])
    # gCurve = np.array([255, 255, 255, 255, 255, 255, 255, 255, 255, 255])
    # gCurve = np.array([128, 128, 128, 128, 128, 128, 128, 128, 128, 128])
    # gCurve = np.array([0, 47, 60, 166, 255])

    # rCurve = np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255 ])
    # rCurve = np.array([0, 0, 0, 25, 45, 70, 100, 125, 220, 255])
    # rCurve = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # rCurve = np.array([255, 255, 255, 255, 255, 255, 255, 255, 255, 255])
    rCurve = np.array([0, 0, 0, 10, 21, 41, 89, 150, 219, 255])

    # rCurve = np.array([0, 3, 53, 162, 255])

    # Create a LookUp Table
    fullRange = np.arange(0, 256)
    rLUT = np.interp(fullRange, originalValue, rCurve)
    bLUT = np.interp(fullRange, originalValue, bCurve)
    gLUT = np.interp(fullRange, originalValue, gCurve)

    # Get the blue channel and apply the mapping
    bChannel = image[:, :, 0]
    bChannel = cv2.LUT(bChannel, bLUT)
    image[:, :, 0] = bChannel

    # Get the green channel and apply the mapping
    gChannel = image[:, :, 1]
    gChannel = cv2.LUT(gChannel, gLUT)
    image[:, :, 2] = gChannel

    # Get the red channel and apply the mapping
    rChannel = image[:, :, 2]
    rChannel = cv2.LUT(rChannel, rLUT)
    image[:, :, 2] = rChannel

    # sharpen = np.array((
    #     [0, -1, 0],
    #     [-1, 5, -1],
    #     [0, -1, 0]), dtype="int")
    # image = cv2.filter2D(image, -1, sharpen)

    url = utils.image_to_url("results/ig_filter_%d.jpg" %
                             (random.randint(0, 10000)), image)
    # cv2.imwrite("results/warmer_%s.jpg" % ('result'), image)
    # return Response('', status=status.HTTP_200_OK)
    return Response({"image_url": url})


@api_view(['POST'])
def ig_filter(request):
    data = utils.parseRequest(request)
    url = data['image_url']
    original = utils.url_to_image(url)
    height, width = original.shape[:2]
    scale = 0.5
    original = cv2.resize(original, (int(width * scale), int(width * scale)))
    image = np.copy(original)

    num_down = 2
    num_bilateral = 7

    for _ in range(num_down):
        image = cv2.pyrDown(image)

    for _ in range(num_bilateral):
        image = cv2.bilateralFilter(image, d=9, sigmaColor=9, sigmaSpace=7)

    for _ in range(num_down):
        image = cv2.pyrUp(image)

    img_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
                              
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB) 
    img_cartoon = cv2.bitwise_and(image, img_edge)
    
    img_cartoon = image_utils.sketchPencilUsingBlending(img_cartoon)

    src_img = cv2.imread('src/sketchpad_texture.jpg')

    # img_cartoon = image_utils.color_transfer(src_img, img_cartoon)
    # img_cartoon = image_utils.color_transfer(img_cartoon, src_img)
    img_cartoon = image_utils.alphablend(src_img, img_cartoon)

    # url = ''
    url = utils.image_to_url("results/ig_filter_%d.jpg" %
                             (random.randint(0, 10000)), img_cartoon)
    return Response({"image_url": url})


@api_view(['POST'])
def sketch(request):
    data = utils.parseRequest(request)
    url = data['src_img']
    image = utils.url_to_image(url)
    sketch = image_utils.sketch(image)
    utils.image_to_url("sketch.jpg",sketch)
    return Response('')
