import cv2
import json
import numpy as np
import dlib
import api_practice.utils as utils
import api_practice.image_utils as image_utils
import api_practice.video_utils as video_utils
import random
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
import api_practice.faceBlendCommon as fbc


PREDICTOR_PATH = 'common/shape_predictor_68_face_landmarks.dat'
faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

@api_view(['POST'])
def face_average(request):
  data = utils.parseRequest(request)

  img1 = utils.url_to_image(data['src_img'])
  img2 = utils.url_to_image(data['dst_img'])
  p1 = fbc.getLandmarks(faceDetector,landmarkDetector, img1)
  p2 = fbc.getLandmarks(faceDetector, landmarkDetector, img2)
  output = image_utils.face_average(img1,img2, p1, p2)
  
  url = utils.image_to_url("results/face_to_average/face_average.jpg", output)
  return Response({"image_url": url})

def face_average2(request):
  data = utils.parseRequest(request)
  image_urls = data['image_urls']
  # image_urls = [data['src_img'], data['dst_img']]

  images = []
  allPoints = []

  for url in image_urls:
    try:
      im = utils.url_to_image(url)
      if im is None:
        print("Unable to load image url")
        next
      else:
        utils.image_to_url("results/face_to_average/%d.jpg" %
                           random.randint(0, 10000), im)
        points = fbc.getLandmarks(faceDetector, landmarkDetector, im)
        if(len(points) > 0):
          allPoints.append(points)
          im = np.float32(im)/255.0
          images.append(im)
        else:
          print("No face detected")
    except:
      print("Forbidden image")
  if len(images) == 0:
    print("No images loaded")
    return Response("no faces to average")

  w = 300
  h = 300

  boundaryPts = fbc.getEightBoundaryPoints(w, h)

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
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

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
  print(output)
  url = utils.image_to_url("results/face_to_average/face_average.jpg", output)
  return Response({"image_url": url})

@api_view(['POST'])
def face_morph(request):
  data = utils.parseRequest(request)
  dst_url = data['dst_url']
  src_url = data['src_url']

  src_img = utils.url_to_image(src_url)
  dst_img = utils.url_to_image(dst_url)

  utils.image_to_url("results/face_morph/morph_orig_src.jpg", src_img)
  utils.image_to_url("results/face_morph/morph_orig_dst.jpg", dst_img)

  src_pts = fbc.getLandmarks(
      faceDetector, landmarkDetector, cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
  dst_pts = fbc.getLandmarks(
      faceDetector, landmarkDetector, cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB))

  src_pts = np.array(src_pts)
  dst_pts = np.array(dst_pts)

  # Convert image to floating point in the range 0 to 1
  src_img = np.float32(src_img)/255.0
  dst_img = np.float32(dst_img)/255.0


  h = 300
  w = 300

  # Normalize image to output coordinates.
  srcNorm, src_pts = fbc.normalizeImagesAndLandmarks((h, w), src_img, src_pts)
  dstNorm, dst_pts = fbc.normalizeImagesAndLandmarks((h, w), dst_img, dst_pts)

  # Calculate average points. Will be used for Delaunay triangulation.
  pointsAvg = (src_pts + dst_pts)/2.0

  # 8 Boundary points for Delaunay Triangulation
  boundaryPoints = fbc.getEightBoundaryPoints(h, w)
  src_pts = np.concatenate((src_pts, boundaryPoints), axis=0)
  dst_pts = np.concatenate((dst_pts, boundaryPoints), axis=0)
  pointsAvg = np.concatenate((pointsAvg, boundaryPoints), axis=0)
  # Calculate Delaunay triangulation.
  rect = (0, 0, w, h)
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

  # Start animation.
  alpha = 0

  while alpha < 1:
    # Compute landmark points based on morphing parameter alpha
    pointsMorph = (1 - alpha) * src_pts + alpha * dst_pts

    # Warp images such that normalized points line up with morphed points.
    imOut1 = fbc.warpImage(srcNorm, src_pts, pointsMorph.tolist(), dt)
    imOut2 = fbc.warpImage(dstNorm, dst_pts, pointsMorph.tolist(), dt)

    # Blend warped images based on morphing parameter alpha
    imMorph = (1 - alpha) * imOut1 + alpha * imOut2

    imMorph = np.uint8(imMorph * 255)
    utils.image_to_url("results/face_morph/morph_%1.2f.jpg" % alpha, imMorph)

    alpha += 0.1
  # imMorph

  return Response("")


@api_view(['POST'])
def video_morph(request):
  data = utils.parseRequest(request)
  dst_url = data['dst_img']
  src_url = data['src_img']

  src_img = utils.url_to_image(src_url)
  dst_img = utils.url_to_image(dst_url)

  utils.image_to_url("results/face_morph/morph_orig_src.jpg", src_img)
  utils.image_to_url("results/face_morph/morph_orig_dst.jpg", dst_img)

  src_pts = fbc.getLandmarks(
      faceDetector, landmarkDetector, cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
  dst_pts = fbc.getLandmarks(
      faceDetector, landmarkDetector, cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB))

  src_pts = np.array(src_pts)
  dst_pts = np.array(dst_pts)

  # Convert image to floating point in the range 0 to 1
  src_img = np.float32(src_img)/255.0
  dst_img = np.float32(dst_img)/255.0

  h = 300
  w = 300

  # Normalize image to output coordinates.
  srcNorm, src_pts = fbc.normalizeImagesAndLandmarks((h, w), src_img, src_pts)
  dstNorm, dst_pts = fbc.normalizeImagesAndLandmarks((h, w), dst_img, dst_pts)

  # Calculate average points. Will be used for Delaunay triangulation.
  pointsAvg = (src_pts + dst_pts)/2.0

  # 8 Boundary points for Delaunay Triangulation
  boundaryPoints = fbc.getEightBoundaryPoints(h, w)
  src_pts = np.concatenate((src_pts, boundaryPoints), axis=0)
  dst_pts = np.concatenate((dst_pts, boundaryPoints), axis=0)
  pointsAvg = np.concatenate((pointsAvg, boundaryPoints), axis=0)
  # Calculate Delaunay triangulation.
  rect = (0, 0, w, h)
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

  # Start animation.
  alpha = 0

  frames = []

  while alpha < 1:
    # Compute landmark points based on morphing parameter alpha
    pointsMorph = (1 - alpha) * src_pts + alpha * dst_pts

    # Warp images such that normalized points line up with morphed points.
    imOut1 = fbc.warpImage(srcNorm, src_pts, pointsMorph.tolist(), dt)
    imOut2 = fbc.warpImage(dstNorm, dst_pts, pointsMorph.tolist(), dt)

    # Blend warped images based on morphing parameter alpha
    imMorph = (1 - alpha) * imOut1 + alpha * imOut2

    imMorph = np.uint8(imMorph * 255)
    utils.image_to_url("results/face_morph/morph_%1.2f.jpg" % alpha, imMorph)
    frames.append(imMorph)

    alpha += 0.05

  path = "results/face_morph/face_average.avi"
  video_utils.video_write(path, 8, (300, 300), frames)
  # video_utils.video_write("results/face_morph/face_average.mp4", 8, (300, 300), frames)
  url = utils.video_to_url(path)
  # return Response("")
  return Response({"video_url": url})


@api_view(['POST'])
def face_swap(request):
  data = utils.parseRequest(request)
  # dst_url = data['dst_img']
  # src_url = data['src_img']

  # Read images
  filename1 = 'src/ted_cruz.jpg'
  filename2 = 'src/donald_trump.jpg'

  img1 = cv2.imread(filename1)
  img2 = cv2.imread(filename2)
  img1Warped = np.copy(img2)

  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

  faceRects = faceDetector(img2, 0)


  frames = []
  frames.append(img2)
  for n in range(0, len(faceRects)):
    # Read array of corresponding points
    points1 = fbc.getLandmarks(
        faceDetector, landmarkDetector, cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    newRect = dlib.rectangle(int(faceRects[n].left()), int(faceRects[n].top()),
                             int(faceRects[n].right()), int(faceRects[n].bottom()))

    points2 = fbc.dlibLandmarksToPoints(landmarkDetector(img2, newRect))

    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    for i in range(0, len(hullIndex)):
      hull1.append(points1[hullIndex[i][0]])
      hull2.append(points2[hullIndex[i][0]])

    # Find delanauy traingulation for convex hull points
    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = fbc.calculateDelaunayTriangles(rect, hull2)

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

    cv2.imwrite("results/face_swap/faceswap_%d.jpg" % n, img2)

    # output = cv2.imread("results/face_swap/faceswap_%d.jpg" % n)
    # output = np.uint8(output)
    # frames.append(output)

  for n in range(0, len(faceRects)):  
    frames.append(cv2.imread("results/face_swap/faceswap_%d.jpg" % n))
    frames.append(cv2.imread("results/face_swap/faceswap_%d.jpg" % n))


  path = "results/face_swap/face_swap.avi"
  video_utils.video_write(path, 1, (1080, 1080), frames)

  return Response('')


@api_view(['POST'])
def video_average(request):
  data = utils.parseRequest(request)
  src_url = data['src_img']
  dst_url = data['dst_img']

  img1 = utils.url_to_image(src_url)
  img2 = utils.url_to_image(dst_url)
  img1Warped = np.copy(img2)

  img2cop = np.copy(img2)
  vh, vw, vc = img2cop.shape
  x_scale = 300/vh
  y_scale = 300/vw
  img2cop = cv2.resize(img2cop, (0, 0), fx=x_scale, fy=y_scale)
  frames = []
  frames.append(img2cop)
  frames.append(img2cop)

  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

  faceRects = faceDetector(img2, 0)
  # print(faceRects)
  points1 = fbc.getLandmarks(faceDetector, landmarkDetector, img1)
  averages = []
  for n in range(0, len(faceRects)):
    newRect = dlib.rectangle(int(faceRects[n].left()), int(faceRects[n].top()),
                             int(faceRects[n].right()), int(faceRects[n].bottom()))
    points2 = fbc.dlibLandmarksToPoints(landmarkDetector(img2, newRect))


    average = image_utils.face_average(img1, img2, points1, points2)
    averages.append(average)

    cv2.imwrite('results/face_swap/average_faces/face_average_%d.jpg' % n, average)

  for k in range(0, len(faceRects)):
    average = averages[k]
    p1 = fbc.getLandmarks(faceDetector, landmarkDetector, average)
    newRect = dlib.rectangle(int(faceRects[k].left()), int(faceRects[k].top()),
                             int(faceRects[k].right()), int(faceRects[k].bottom()))
    p2 = fbc.dlibLandmarksToPoints(landmarkDetector(img2, newRect))
    frame = image_utils.face_swap(average, img2, p1, p2)

    # frames.append(frame)
    # frames.append(frame)
    vid_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vh, vw, vc = vid_frame.shape
    x_scale = 300/vh
    y_scale = 300/vw

    vid_frame = cv2.resize(vid_frame, (0,0), fx = x_scale, fy = y_scale)
    frames.append(vid_frame)
    frames.append(vid_frame)

    img2 = frame

    cv2.imwrite('results/face_swap/frames/frame_%d.jpg' % k, vid_frame)

  path = 'results/face_swap/face_swap.avi'
  h, w, c = img2.shape
  video_utils.video_write(path, 2, (300, 300), frames)

  # video_utils.video_write(path, 2, (w, h), frames)
  url = utils.video_to_url(path)
  return Response({"video_url": url})


@api_view(['POST'])
def test(request):
  data = utils.parseRequest(request)
  src_url = data['src_img']
  img1 = utils.url_to_image(src_url)

  dst = image_utils.brighnessAndContrastAuto(img1, img1, 0.5)
  dst = image_utils.cool(dst)
  cv2.imwrite("hello.jpg", dst)

  dst = image_utils.cool(img1)
  dst = image_utils.brighnessAndContrastAuto(dst, dst, 0.5)
  cv2.imwrite("hello2.jpg", dst)

  dst = image_utils.cool(img1)
  cv2.imwrite("hello3.jpg", dst)

  # dst = image_utils.cool(dst)
  dst = image_utils.brighnessAndContrastAuto(img1, img1, 1)
  cv2.imwrite("hello4.jpg", dst)

  dst = image_utils.brighnessAndContrastAuto(img1, img1, 0.5)
  dst = image_utils.warm(dst)
  cv2.imwrite("hello5.jpg", dst)

  dst = image_utils.warm(img1)
  dst = image_utils.brighnessAndContrastAuto(dst, dst, 0.5)
  cv2.imwrite("hello6.jpg", dst)

  dst = image_utils.warm(img1)
  cv2.imwrite("hello7.jpg", dst)

  return Response('')

@api_view(['POST'])
def sketch(request):
  data = utils.parseRequest(request)
  url = data['src_img']
  image = utils.url_to_image(url)
  h, w, c = image.shape
  frames = []
  frames2 = []
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  inverted = 255 - gray
  frames.append(image)
  # frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
  # frames.append(cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR))
  frames2.append(image)
  # frames2.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
  # frames2.append(cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR))
  
  # canvas = cv2.imread("src/paper-texture.jpg", 0)
  canvas = cv2.imread("src/sketchpad-texture.jpg", 0)

  h, w, c = image.shape
  canvas = cv2.resize(canvas, (w, h))

  image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)

  for i in range(1, random.randint(15, 25)):
    sketch = image_utils.sketch(image, i, canvas)
    sketch2 = image_utils.sketch2(image, i, canvas)
    # utils.image_to_url("results/sketch/s_%d.jpg" % i, sketch)
    # sketch = cv2.stylization(sketch, sigma_s=60, sigma_r=0.07)
    # sketch2 = cv2.stylization(sketch, sigma_s=60, sigma_r=0.07)

    frames.append(sketch)
    frames2.append(sketch2)
  video_utils.video_write("results/sketch/sketch.mp4", 4, (w, h), frames)
  video_utils.video_write("results/sketch/sketch2.mp4", 4, (w, h), frames2)

  # video_utils.video_write("results/sketch/sketch.avi", 2, (w, h), frames)
  return Response('')
