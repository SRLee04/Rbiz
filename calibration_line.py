# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:54:56 2021

@author: 82106
"""

# 로봇으로 인식 된 물체로 이동 시켜 로봇 위치 좌표를 측정 한다.
# 물체 3개에 대한 좌표를 추출한 후에 dx,dy에 넣는다.
#이미지 행렬 [x,y,1]
import numpy as np
import cv2
import matplotlib.pyplot as plt


#### 리스트 나누기 함수
def list_div(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

#### 영상 출력
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)


def captuer(event, x, y, flags, params):
    if event==cv2.EVENT_LBUTTONDOWN:
        print("capture")
        cv2.imwrite('calibration_line.jpg',src_bin)

cv2.namedWindow('VideoFrame', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("VideoFrame", captuer)

while True:
    ret, frame = capture.read()
    # x=80 ,y =87
    # x2=465 , y2=345
    line = frame[87:345, 80:465]  #[y:y2,x:x2]
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)

    th, src_bin = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    src_bin = cv2.bitwise_not(src_bin) #흑백 반전

    cv2.namedWindow('line', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("line", src_bin)

    cv2.namedWindow('VideoFrame', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("VideoFrame", frame)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

cv2.waitKey()
cv2.destroyAllWindows()

## 캡쳐 이미지 호출

img = cv2.imread("calibration_line.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_blurred=cv2.GaussianBlur(img_gray,ksize=(7,7),sigmaX=0) # 이미지 노이지 줄이는 역할
th, img_thres = cv2.threshold(img_blurred, 150, 255, cv2.THRESH_BINARY) # 이미지 임계를 설정

src_bin = cv2.bitwise_not(img_thres) #흑/백 반전
contours, _ = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

point=[]

for cnt in contours:

    # 너무 작은 객체는 제외
    # if cv2.contourArea(cnt) < 1500 or cv2.contourArea(cnt)> 3000:
    #     continue
    cv2.polylines(img,[cnt],True,(0,0,0),1)

    # get rectangel

    rect=cv2.minAreaRect(cnt)

    (x,y),(w,h),angel = rect

    #Disply Rectangel
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    point.append(x)
    point.append(y)

    # 이미지 용
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter([x], [y], c="r", s=20)
    plt.text(x+10, y+10, f'({int(x)},{int(y)})',bbox={'boxstyle': 'round', 'ec': (1.0, 0.5, 0.5), 'fc': (1.0, 0.8, 0.8)})

plt.show()

# list_test = list(range(1,32))
# print("분할 전 : ", list_test)

point_div = list_div(point, 2)
print("분할 후 : ", point_div)

### calibration 좌표

x1=point_div[0][0]
y1=point_div[0][1]

x2=point_div[1][0]
y2=point_div[1][1]

x3=point_div[2][0]
y3=point_div[2][1]

####calibration

def calibration_line(x,y):

    ##robot 좌표

        ### dx 행렬
    X=np.mat([[-0.252997918],
          [0.329915104],
          [0]])


        ### dy 행렬
    Y=np.mat([[0.021884545],
          [-0.014027422],
          [0]])


    ### 이미지 행렬
    Prime=np.mat([[x1, y1, 1],
              [x2, y2, 1],
              [x3, y3, 1]])


    ### dx, dy 행렬
    dx_matrix = Prime.I*X
    dy_matrix = Prime.I*Y

    print("Real_X=")
    print(dx_matrix)
    print("a_x= ", dx_matrix[0, 0])
    print("b_x= ", dx_matrix[1, 0])
    print("c_x= ", dx_matrix[2, 0])
    print("Real_Y=")
    print("a_y= ", dy_matrix[0, 0])
    print("b_y= ", dy_matrix[1, 0])
    print("c_y= ", dy_matrix[2, 0])
    print(dy_matrix)



    image_x = x


    image_y = y


    r_x =-0


    r_y = 0


    ### Calibration 식
    dx = dx_matrix[0, 0]*image_x + dx_matrix[1, 0]*image_y + dx_matrix[2, 0]
    dy = dy_matrix[0, 0]*image_x + dy_matrix[1, 0]*image_y + dy_matrix[2, 0]



    print(r_x + dx)
    print(r_y + dy)

calibration_line(1, 2)  # x,y 값 넣기

