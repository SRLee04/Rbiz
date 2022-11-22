import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import math
from operator import itemgetter

# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
dev = rs.device()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
depth_frame = None
color_intrin = None
depth_intrin = None
depth_to_color_extrin = None

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()


# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

while True:
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue

    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)

    depth_frame = aligned_depth_frame.get_data()
    depth_image = np.asanyarray(depth_frame)
    color_image = np.asanyarray(color_frame.get_data())

####################################

    img = color_image
    img_resize = cv2.resize(img,(640,480))
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    img_blurred=cv2.GaussianBlur(img_gray,ksize=(7,7),sigmaX=0) # 이미지 노이지 줄이는 역할
    th, img_thres = cv2.threshold(img_blurred, 150, 255, cv2.THRESH_BINARY) # 이미지 임계를 설정

    # src_bin = cv2.bitwise_not(img_thres) #흑/백 반전
    contours, _ = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # 중심점 추가

    point=[]
    for cnt in contours:

        # 너무 작은 객체는 제외
        if cv2.contourArea(cnt) < 2000 or cv2.contourArea(cnt)> 4000:
            continue
        cv2.polylines(img_resize,[cnt],True,(0,0,255),1)

        # get rectangel

        rect=cv2.minAreaRect(cnt)

        (x,y),(w,h),angel = rect

        #Disply Rectangel
        box=cv2.boxPoints(rect)
        box=np.int0(box)
        point.append(box)
        cv2.circle(img_resize, (int(x), int(y)), 4, (0, 0, 255), -1)

        # 높이 측정

        dist = aligned_depth_frame.get_distance(int(x), int(y))
        dist = dist*100

        cv2.putText(img_resize, f'({round(x)} , {round(y)}, {round(dist)}cm)', (int(x),int(y)), cv2.FONT_HERSHEY_PLAIN, 1,(197, 65, 217), 1)



    # plt.show()
    ret = np.array(point)
    points=ret.tolist() # 리스트 변환

    print(points)

    for i in points:
        for k in i[:]:
            if k[1]<y:
                i.remove(k) # 중심점 이하 y 값의 리스트 삭제


    # print(points)


    # 각도 추가

    angelbtw=[]

    for i in points:

        m1=0 #시작 좌표

        m2= abs((i[0][1]-i[1][1])/(i[0][0]-i[1][0])) # 직선
        v=abs((m1 - m2) / (1 + m1 * m2))

        angelbtw.append({'x':(i[0][0]+i[1][0])/2,'y':(i[0][1]+i[1][1])/2,'anglediff': int(np.degrees(np.arctan(v)))})

    print(angelbtw)
    # for angel in angelbtw:
    #     x=angel['x']
    #     y=angel['y']
    #     ang=angel['anglediff']
    #
    #
    #     plt.imshow(cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB))
    #     plt.scatter([x], [y], c="r", s=30)
    #     plt.text(x, y, f'{ang}°',horizontalalignment='center')

    # plt.show()

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', img_resize)
    key = cv2.waitKey(1)



    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
