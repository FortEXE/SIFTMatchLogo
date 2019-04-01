# nama: m. adnan khairi as.
# nim: 1603786
# email: adnankhairi@student.upi.edu
# python SIFT object logo detector

import cv2
import numpy as np
#from matplotlib import pyplot as plt

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

img1 = cv2.imread('ax.png',0)
img2 = cv2.VideoCapture('vidcut2.mp4',0)

#deteksi dan buat descriptornya
#keypoint 1 dan descriptor 1 dengan sift
kp1, des1 = sift.detectAndCompute(img1,None)
i = 0

while(img2.isOpened()):
    ret, frame2 = img2.read()

    #keypoint 2 dan descriptor 2 dengan sift
    kp2, des2 = sift.detectAndCompute(frame2,None)
    
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    #BFMatcher dengan parameter default
    if (des2 is not None):
        matches = bf.match(des1,des2)
    else:
        break
    
    #mendapatkan matches yang telah disort
    matches = sorted(matches, key = lambda x:x.distance)

    #memilih 1000 titik paling baik
    good_match = matches[:1000]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    matchesMask = mask.ravel().tolist()
    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    dst = cv2.perspectiveTransform(pts,M)
    #dst += (w, 0)  # adding offset

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatches(img1,kp1,gray,kp2,good_match,None,**draw_params) 
    
    #cv2.imshow('result', img3)
    #print(dst[1][0][1], "\n", dst[3][0][1], "\n\n\n")

    #pembatasan untuk membuat persegi dan trapezium untuk deteksi logo yang ditemukan
    if (((dst[0][0][1]) < (dst[2][0][1]) and (dst[1][0][1]) > (dst[3][0][1])) and ((dst[0][0][0]) < (dst[2][0][0]) and (dst[1][0][0]) < (dst[3][0][0]))) :
        frame = cv2.polylines(frame2,[np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)
        i += 1
        

    #tampilan video
    cv2.imshow('',frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('frame logo yang ada pada video = ', i)
img2.release()
cv2.destroyAllWindows()



#deteksi keypoint
#kp = sift.detect(gray,None)
#cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('keypoints', img)
#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)









