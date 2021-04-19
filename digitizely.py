import numpy as np
import cv2
import math
import cairo
from numpy.linalg import norm

def getLines(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    maxBrightness = gray.max()
    gray += (255-maxBrightness)
    edges = cv2.Canny(gray,100,255,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)

    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
        
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imwrite('houghlines3.jpg',img)
    return lines

def getCorners(image):
  # convert the image to grayscale, blur it, and find edges
  # in the image
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (5, 5), 0)
  edged = cv2.Canny(gray, 75, 200)

  # show the original image and the edge detected image
  print "STEP 1: Edge Detection"
  cv2.imshow("Image", image)
  cv2.imshow("Edged", edged)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # find the contours in the edged image, keeping only the
  # largest ones, and initialize the screen contour
  cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
  cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
  # loop over the contours
  for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
      screenCnt = approx
      break
 
  # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return screenCnt

def get_dist(s,p):
  return  math.sqrt(math.pow(s[0]-p[0],2)+math.pow(s[1]-p[1],2))

#[(582, 1769), (1537,1452), (2484,2231), (1263, 2814)]
def order_points(pts):
  # initialzie a list of coordinates that will be ordered
  # such that the first entry in the list is the top-left,
  # the second entry is the top-right, the third is the
  # bottom-right, and the fourth is the bottom-left
  rect = np.zeros((4, 2), dtype = "float32")
 
  # the top-left point will have the smallest sum, whereas
  # the bottom-right point will have the largest sum
  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]

  # now, compute the difference between the points, the
  # top-right point will have the smallest difference,
  # whereas the bottom-left will have the largest difference
  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(abs(diff))]
  pts.remove(rect[1])
  rect[3] = pts[np.argmax(abs(diff))]
  pts.remove(rect[3])

  # return the ordered coordinates
  return rect
 
def four_point_transform(image, pts):
  # obtain a consistent order of the points and unpack them                      # individually                                                                
  rect = pts
  (tl, tr, br, bl) = rect

  # compute the width of the new image, which will be the                       
  # maximum distance between bottom-right and bottom-left                       
  # x-coordiates or the top-right and top-left x-coordinates                    
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))

  # compute the height of the new image, which will be the                      
  # maximum distance between the top-right and bottom-right                     
  # y-coordinates or the top-left and bottom-left y-coordinates                 
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))


  # now that we have the dimensions of the new image, construct                 
  # the set of destination points to obtain a "birds eye view",                 
  # (i.e. top-down view) of the image, again specifying points                  
  # in the top-left, top-right, bottom-right, and bottom-left                   
  # order                                                                       
  dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype = "float32")

  # compute the perspective transform matrix and then apply it                  
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

  # return the warped image                                                     
  return warped
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = 1440
 
  # compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = 900
 
  # now that we have the dimensions of the new image, construct
  # the set of destination points to obtain a "birds eye view",
  # (i.e. top-down view) of the image, again specifying points
  # in the top-left, top-right, bottom-right, and bottom-left
  # order
  dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype = "float32")
 
  # compute the perspective transform matrix and then apply it
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
  # return the warped image
  return warped

print(cv2.__version__)

im = cv2.imread('img.png')


lines = getLines(im)

#array containing 4 points
#pts = np.array(eval(args["coords"]), dtype = "float32")
#pts = np.array([[515, 86], [984,87], [977,541], [525, 544]], dtype = "float32")
 
#pts = getCorners(im)

# apply the four point tranform to obtain a "birds eye view" of
# the image
#warped = four_point_transform(im, pts)

#cv2.imshow('image', warped)
#cv2.waitKey(0)

# grab the dimensions of the image and calculate the center
# of the image
(h, w) = im.shape[:2]
center = (w / 2, h / 2)

#convert image to b&w
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#normalize brightness
#find brightest pixels and difference between that and white
#add difference to every pixel
maxBrightness = imgray.max()
imgray += (255-maxBrightness)

#blur
#blur = cv2.blur(flipped,(3,3))

#thresholding
(thresh, binRed) = cv2.threshold(imgray, 160, 255, cv2.THRESH_BINARY)

#simple thresholding
_, Rcontours, hier_r = cv2.findContours(binRed,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
r_areas = [cv2.contourArea(c) for c in Rcontours]
max_rarea = np.argmax(r_areas)
CntExternalMask = np.ones(binRed.shape[:2], dtype="uint8") * 255

contourMax= Rcontours[max_rarea]

# calc arclentgh
arclen = cv2.arcLength(contourMax, True)

# approx the contour
epsilon = arclen * 0.0005

#epsilon = arclen * 0.0001
contour = cv2.approxPolyDP(contourMax, epsilon, True)

#cairo code
surface = cairo.PDFSurface("output.pdf", w, h)
context = cairo.Context(surface)

context.set_line_width(0.9)
context.move_to(contour[0][0][0],contour[0][0][1])

print("contour size")
print(len(contour))
print(len(lines))
print(len(lines[0]))
#change to index iteration
for ii in range(1, len(contour)-1):

  #look two points in advance
  for jj in range(ii, ii+2):

    #edge case for when jj is outside of array
    if jj >= len(contour):
      context.line_to(contour[ii][0][0], contour[ii][0][1])
      break

    #if any of those points are on a line, line to
    #else curve to the third point away
    for rho,theta in lines[0]:
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1000*(-b))
      y1 = int(y0 + 1000*(a))
      x2 = int(x0 - 1000*(-b))
      y2 = int(y0 - 1000*(a))

      p1 = (x1, y1)
      p2 = (x2, y2)
      p3 = (contour[jj][0][0], contour[jj][0][1])
      numerator = norm(np.cross(np.subtract(p2, p1), np.subtract(p1, p3)))
      denominator = norm(np.subtract(p2, p1))
      xDistance = np.divide(numerator, denominator) #distance between point and line

      if xDistance < 5:
        context.line_to(contour[ii][0][0], contour[ii][0][1])
        print("line")
        jj = ii + 2
      elif jj == ii+1 and ii+2 < len(contour):
        context.curve_to(contour[ii][0][0], contour[ii][0][1], contour[ii+1][0][0], contour[ii+1][0][1], contour[ii+2][0][0], contour[ii+2][0][1])
        print("curve")
        jj = ii + 2
        #advance marker to the point that was just curved to, omitting the points in between
        ii = ii + 2
        
context.close_path()

context.stroke()

surface.finish()

