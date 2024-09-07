import numpy as np
import cv2

from hed import deepEdges

def dominantRectangle(frame):

    out = deepEdges(frame)

    _, edges = cv2.threshold(out,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    center = np.array([frame.shape[1]/2,frame.shape[0]/2])

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contour, indicators in zip(contours, hierarchy[0]):
        if indicators[2] == -1 and indicators[3] > 0:
            peri = cv2.arcLength(contour, True)
            eps = 0.08  
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if cv2.pointPolygonTest(approx, center, False) > 0:
                if cv2.contourArea(approx) > 1000:
                    if len(approx) == 4:
                        if cv2.isContourConvex(approx):
                            lengths = []
                            for p, q in zip(approx,np.roll(approx,-2)):
                                length = np.linalg.norm(np.array(p)-np.array(q))
                                lengths.append(length)
                            threshold = 32.0
                            if abs(lengths[0]-lengths[2]) < threshold and abs(lengths[1]-lengths[3]) < threshold:
                                rects.append(approx.squeeze(1))
                            #else:
                            #    print('threshold failed',abs(lengths[0]-lengths[2]),abs(lengths[1]-lengths[3]))
                        #else:
                        #    print('not convex')
                    #else:
                    #    print('approx',len(approx))

    if rects:
        distances = []
        for rect in rects:
            centroid = np.array(rect).mean(axis=0)
            distance = np.linalg.norm(centroid-center)
            distances.append(distance)

        central_rect = rects[np.argmin(distances)]
        return central_rect
    
    return None

def drawRectangle(frame,central_rect):
    for p, q in zip(central_rect,np.roll(central_rect,-2)):
        cv2.line(frame,p,q,(0,255,0),1)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum, bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # The top-right point will have the smallest difference, bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect
    
def extractRectangle(image, pts):
    # Order the points in a consistent manner
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate the width of the new image, which will be the maximum distance between
    # bottom-right and bottom-left or top-right and top-left
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    # Calculate the height of the new image, which will be the maximum distance between
    # top-right and bottom-right or top-left and bottom-left
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Construct the destination points which will be a rectangle
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
       
    return warped
    
if __name__ == '__main__':
    frame = cv2.imread('1724513323.png')
    central_rect = dominantRectangle(frame)
    if central_rect is not None:
        drawRectangle(frame,central_rect)
    cv2.imwrite('dominant_rect.png',frame)
