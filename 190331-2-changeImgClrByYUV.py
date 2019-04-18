import cv2
img = cv2.imread('lena.png')
yuv_img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow('yuv_img',yuv_img)
key=cv2.waitKey()
if key==27:
    cv2.destroyAllWindows()



rgb_img = cv2.cvtColor(yuv_img,cv2.COLOR_YUV2BGR)
cv2.imshow('rgb_img',rgb_img)
key=cv2.waitKey()
if key==27:
    cv2.destroyAllWindows()