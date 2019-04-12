import cv2
import numpy as np
import api_practice.faceBlendCommon as fbc
from pyhull.delaunay import DelaunayTri
from pyhull.convex_hull import ConvexHull
import scipy.ndimage


def loadImageYcb(image):
    ycbImage = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycbImage = np.float32(ycbImage)
    return ycbImage

def colorDodge(top, bottom):

  # divid the bottom by inverted top image and scale back to 250
  output = cv2.divide(bottom, 255 - top, scale=256)

  return output

def sketchPencilUsingBlending(original, kernelSize=21):
  img = np.copy(original)

  # Convert to grayscale
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Invert the grayscale image
  imgGrayInv = 255 - imgGray

  # Apply GaussianBlur
  imgGrayInvBlur = cv2.GaussianBlur(imgGrayInv, (kernelSize, kernelSize), 0)

  # blend using color dodge
  output = colorDodge(imgGrayInvBlur, imgGray)

  return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

def color_transfer(src, dst):
  output = np.copy(dst)
  srcLab = np.float32(cv2.cvtColor(src, cv2.COLOR_BGR2LAB))
  dstLab = np.float32(cv2.cvtColor(dst, cv2.COLOR_BGR2LAB))
  outputLab = np.float32(cv2.cvtColor(output, cv2.COLOR_BGR2LAB))

  print(src)

  print(srcLab)

  # Split the Lab images into their channels
  srcL, srcA, srcB = cv2.split(srcLab)
  dstL, dstA, dstB = cv2.split(dstLab)
  outL, outA, outB = cv2.split(outputLab)
  

  outL = dstL - dstL.mean()
  outA = dstA - dstA.mean()
  outB = dstB - dstB.mean()

  # scale the standard deviation of the destination image
  outL *= srcL.std() / (dstL.std() if dstL.std() else 1)
  outA *= srcA.std() / (dstA.std() if dstA.std() else 1)
  outB *= srcB.std() / (dstB.std() if dstB.std() else 1)

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
  #  cv2.imwrite("results/oldify_%d.jpg" % (random.randint(0, 10000)), output)
  return output

def alphablend(background, foreground):

  b, g, r, a = cv2.split(foreground)

  # Save the foregroung RGB content into a single object
  foreground = cv2.merge((b, g, r))

  # Save the alpha information into a single Mat
  alpha = cv2.merge((a, a, a))

  # Read background image
  background = cv2.imread("../data/images/backGroundLarge.jpg")

  # Convert uint8 to float
  foreground = foreground.astype(float)
  background = background.astype(float)
  alpha = alpha.astype(float)/255

  # Perform alpha blending
  foreground = cv2.multiply(alpha, foreground)
  background = cv2.multiply(1.0 - alpha, background)
  outImage = cv2.add(foreground, background)
  return outImage

def face_swap(img1, img2, points1, points2):
  img1Warped = np.copy(img2)

  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
  # Find convex hull
  hull1 = []
  hull2 = []

  # hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
  hullIndex = ConvexHull(np.array(points2)).vertices


  for i in range(0, len(hullIndex)):
    hull1.append(points1[hullIndex[i][0]])
    hull2.append(points2[hullIndex[i][0]])



  # Find delanauy traingulation for convex hull points
  sizeImg2 = img2.shape
  rect = (0, 0, sizeImg2[1], sizeImg2[0])

  # dt = fbc.calculateDelaunayTriangles(rect, hull2)
  dt = DelaunayTri(hull2, True).vertices
  # dt = Dela

  if len(dt) == 0:
    quit()

  # Apply affine transformation to Delaunay triangles
  for i in range(0, len(dt)):
    t1 = []
    t2 = []

    #get points for img1, img2 corresponding to the triangles
    for j in range(0, 3):
      t1.append(hull1[dt[i][j]])
      t2.append(hull2[dt[i][j]])

    fbc.warpTriangle(img1, img1Warped, t1, t2)

  # Calculate Mask for Seamless cloning
  hull8U = []
  for i in range(0, len(hull2)):
    hull8U.append((hull2[i][0], hull2[i][1]))

  mask = np.zeros(img2.shape, dtype=img2.dtype)

  cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
  # find center of the mask to be cloned with the destination image
  r = cv2.boundingRect(np.float32([hull2]))

  center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

  # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

  # Clone seamlessly.
  img2 = cv2.seamlessClone(
      np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
    
  return img2

def face_average(img1, img2, points1, points2):
  images = []
  allPoints = []
  allPoints.append(points1)
  allPoints.append(points2)
  images.append(np.float32(img1)/255.0)
  images.append(np.float32(img2)/255.0)

  w = 300
  h = 300

  boundaryPts = fbc.getEightBoundaryPoints(h, w)

  numImages = len(images)
  numLandmarks = len(allPoints[0])


  imagesNorm = []
  pointsNorm = []

  pointsAvg = np.zeros((numLandmarks, 2), dtype=np.float32)
  # Warp images and trasnform landmarks to output coordinate system,
  # and find average of transformed landmarks.
  for i, img in enumerate(images):
    points = allPoints[i]
    points = np.array(points)

    img, points = fbc.normalizeImagesAndLandmarks((h, w), img, points)


    # Calculate average landmark locations
    pointsAvg = pointsAvg + (points / (1.0*numImages))

    # Append boundary points. Will be used in Delaunay Triangulation
    points = np.concatenate((points, boundaryPts), axis=0)

    pointsNorm.append(points)
    imagesNorm.append(img)

  # Append boundary points to average points.
  pointsAvg = np.concatenate((pointsAvg, boundaryPts), axis=0)

  # Delaunay triangulation
  rect = (0, 0, w, h)
  # dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)
  dt = DelaunayTri(pointsAvg, True).vertices


  # Output image
  output = np.zeros((h, w, 3), dtype=np.float)

  # Warp input images to average image landmarks
  for i in range(0, numImages):

    imWarp = fbc.warpImage(
        imagesNorm[i], pointsNorm[i], pointsAvg.tolist(), dt)

    # Add image intensities for averaging
    output = output + imWarp

  # Divide by numImages to get average
  output = output / (1.0*numImages)
  output = output * 255.0
  output = np.uint8(output)
  # print(output)
  return output


# **
#  *  \brief Automatic brightness and contrast optimization with optional histogram clipping
#  *  \param [in]src Input image GRAY or BGR or BGRA
#  *  \param [out]dst Destination image 
#  *  \param clipHistPercent cut wings of histogram at given percent tipical=>1, 0=>Disabled
#  *  \note In case of BGRA image, we won't touch the transparency
# *
def brighnessAndContrastAuto(src, dst, clipHistPercent=0):
  histSize = 256
  alpha = 0
  beta = 0
  minGray = 0
  maxGray = 0

  # //to calculate grayscale histogram

  gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  cv2.imwrite('gray.jpg', gray)

  # gray = np.float32(gray)
  if clipHistPercent == 0:
    minGray, maxGray = cv2.minMaxLoc(src)
  else:
    uniform = True
    accumulate = True
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256], True, False)
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX, -1)
    # hist = np.histogram(gr)
    # print(hist)
    accumulator = []
    accumulator.append(hist[0])
    print(accumulator)
    for i in range(1, histSize - 1):
      # accumulator[i] = accumulator[i - 1] + hist[i]
      accumulator.append(accumulator[i - 1] + hist[i])
    max = accumulator[len(accumulator) - 1]

    clipHistPercent = clipHistPercent * (max / 100.0)

    clipHistPercent = clipHistPercent / 2
    minGray = 0
    # print(accumulator[minGray][0] < clipHistPercent[0])
    while accumulator[minGray][0] < clipHistPercent[0]:
      minGray = minGray + 1
    # for i in range(0, histSize - 2):
      # print(accumulator[i])
    
    maxGray = histSize - 1 - 1
    while accumulator[maxGray][0] >= max - clipHistPercent[0]:
      maxGray = maxGray - 1


  inputRange = maxGray - minGray

  alpha = (histSize - 1)/inputRange
  beta = -minGray * alpha

  # print(minGray)
  # print(maxGray)
  print(alpha)
  print(beta)

  # dst = np.array([])
  # print(dst)
  # cv2.convertScaleAbs(src, dst, alpha, beta)
  dst = cv2.convertScaleAbs(src, alpha = alpha, beta = beta)
  # print(dst)
  # cv2.imwrite('hello.jpg', dst)

  # // restore alpha channel from source 
  # if (dst.type() == CV_8UC4)
  # {
  #     int from_to[] = { 3, 3};
  #     cv::mixChannels(&src, 4, &dst,1, from_to, 1);
  # }
  return dst


def cool(original):

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
  return image

def warm(original):

  image = np.copy(original)

  # Pivot points for X-Coordinates
  originalValue = np.array([0, 50, 100, 150, 200, 255])
  # Changed points on Y-axis for each channel
  # bCurve = np.array([0, 80 + 35, 150 + 35, 190 + 35, 220 + 35, 255])
  # rCurve = np.array([0, 20 - 20,  40 - 20,  75 - 20, 150 + 20, 255])
  rCurve = np.array([0, 80, 150, 190, 220, 255])
  bCurve = np.array([0, 20,  40,  75, 150, 255])

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
  return image


def sketch(image, sigma = 5, canvas = None):
  img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  img_blur = cv2.GaussianBlur(img_gray, (0,0), sigma, sigma)
  img_blend = cv2.divide(img_gray, img_blur, scale=256)
  if canvas is not None:
    img_blend = cv2.multiply(img_blend, canvas, scale=1./256)
  return cv2.cvtColor(img_blend, cv2.COLOR_GRAY2BGR)

def sketch2(image, sigma = 5, canvas = None):
  gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
  inverted = 255 - gray
  blur = scipy.ndimage.filters.gaussian_filter(inverted, sigma)
  def dodge(front, back):
    try:
        result = front*255/(255-back)
    except ZeroDivisionError:
        result = 0
    result[np.logical_or(result > 255, back == 255)] = 255
    return np.uint8(result)
  blend = dodge(blur, gray)
  if canvas is not None:
    # blend = cv2.multiply(blend, canvas, scale = 1./256)
    blend = cv2.multiply(canvas, blend, scale=1./256)

  return cv2.cvtColor(blend, cv2.COLOR_GRAY2BGR)
