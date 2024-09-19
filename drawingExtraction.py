import numpy as np
import cv2
import random
import time

def huang_thresholding(image):
    # Compute the histogram
    hist, bin_edges = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Normalize the histogram
    hist = hist.astype(np.float32)
    hist /= hist.sum()

    # Compute the cumulative histogram
    cum_hist = np.cumsum(hist)
    mean = np.cumsum(hist * np.arange(256))

    # Calculate the threshold using Huang's method
    threshold = -1
    min_entropy = np.inf

    for t in range(1, 256):
        p1 = cum_hist[t]
        p2 = 1 - p1

        if p1 == 0 or p2 == 0:
            continue

        mean1 = mean[t] / p1
        mean2 = (mean[-1] - mean[t]) / p2

        entropy1 = np.log(mean1) if mean1 > 0 else 0
        entropy2 = np.log(mean2) if mean2 > 0 else 0

        total_entropy = p1 * entropy1 + p2 * entropy2

        if total_entropy < min_entropy:
            min_entropy = total_entropy
            threshold = t

    return threshold
    
def get_trajectories(skeleton_image):
    trajectories = []
    contours, _ = cv2.findContours(skeleton_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) > 10:
            visited = set()
            trajectory = []
            for countour_point in contour:
                point = tuple(countour_point[0])
                if point in visited:
                    if len(trajectory) > 10:
                        trajectories.append(trajectory)
                    trajectory = []
                else:
                    trajectory.append(point)
                    visited.add(point)
            if trajectory:
                if len(trajectory) > 10:
                    trajectories.append(trajectory)
    
    return trajectories

def extract_trajectories(img):
    # Apply Huang thresholding
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    threshold = huang_thresholding(img)
    threshold = int(0.97*threshold)
    _, binary = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    drawing_percentage = (binary.size - cv2.countNonZero(binary)) / binary.size
    if drawing_percentage > 0.1:
        return []
    cv2.imwrite(f'logs/binary{int(time.time())}.png',binary)

    # Erode the image using a 5x5 structural element
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    #cv2.imwrite('eroded.png',eroded)

    # Fill the components that are touching the border with white (255)
    h, w = eroded.shape
    mask = np.zeros((h+2, w+2), np.uint8) 
    cv2.floodFill(eroded, mask, (0, 0), 255)
    cv2.floodFill(eroded, mask, (w-1, 0), 255)
    cv2.floodFill(eroded, mask, (0, h-1), 255)
    cv2.floodFill(eroded, mask, (w-1, h-1), 255)
    #cv2.imwrite('filled.png',eroded)

    # Perform skeletonization on the remaining areas
    skeleton = cv2.ximgproc.thinning(~eroded)
    cv2.imwrite(f'logs/skeleton{int(time.time())}.png',skeleton)

    # Turn skeleton into trajectories
    trajectories = get_trajectories(skeleton)
    
    return trajectories # list of list of points

def visualize_trajectories(trajectories, image_shape):
    # Create a blank image to draw the trajectories on
    output_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    for trajectory in trajectories:
        # Randomly choose a color for each trajectory
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Draw the trajectory on the image
        for i in range(len(trajectory) - 1):
            pt1 = (trajectory[i][0], trajectory[i][1])  # (x, y) format
            pt2 = (trajectory[i+1][0], trajectory[i+1][1])  # (x, y) format
            cv2.line(output_image, pt1, pt2, color, thickness=1)

    return output_image
    
if __name__ == '__main__':
    img = cv2.imread('img.png')
    trajectories = extract_trajectories(img)
    result = visualize_trajectories(trajectories, img.shape)
    cv2.imwrite('result.png',result)
