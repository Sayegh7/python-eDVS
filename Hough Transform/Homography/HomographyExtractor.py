import cv2
import numpy as np
 
if __name__ == '__main__' :
 
    # Read source image.
    im_src = cv2.imread('botView.jpg')
    # Four corners of the book in source image
    pts_src = np.array([[44, 101], [90, 101], [108, 123],[34, 122]])
 
 
    # Read destination image.
    im_dst = cv2.imread('topView.jpg')
    # Four corners of the book in destination image.
    pts_dst = np.array([[33, 5],[100, 4],[101, 85],[33, 84]])
 
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    print h 
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    print im_out.shape
    print im_out
     
    # Display images
#    cv2.imshow("Source Image", im_src)
#    cv2.imshow("Destination Image", im_dst)
#    cv2.imshow("Warped Source Image", im_out)
    cv2.imwrite("warpView.jpg", im_out)
    cv2.waitKey(0)