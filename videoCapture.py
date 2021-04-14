import cv2
 
# 불러올 영상 경로, 파일명
vidcap = cv2.VideoCapture('videos/test2.mp4')
 
count = 0
 
while(vidcap.isOpened()):

    ret, image = vidcap.read()
 
    #캡쳐된 이미지 저장
    #프레임단위 설정
    if(int(vidcap.get(1)) % 10 == 0):
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        cv2.imwrite("cutimgs/frame%d.jpg" % count, image)
        print('Saved frame%d.jpg' % count)
        count += 1

vidcap.release()

