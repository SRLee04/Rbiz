import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import socket

#### 통신 추가 10/24
#서버의 주소입니다. hostname 또는 ip address를 사용할 수 있습니다.
# HOST = '192.168.0.23'
# # 서버에서 지정해 놓은 포트 번호입니다.
# PORT = 9999
#
# # 소켓 객체를 생성합니다.
# # 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용합니다.
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
#
# # 지정한 HOST와 PORT를 사용하여 서버에 접속합니다.
# client_socket.connect((HOST, PORT))


# # 영상 이미지 캡쳐
# capture = cv2.VideoCapture(1)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
#
#
# def captuer(event, x, y, flags, params):
#     if event==cv2.EVENT_LBUTTONDOWN:
#         print("capture")
#         cv2.imwrite('linepoint.jpg',src_bin)
#
# cv2.namedWindow('VideoFrame', cv2.WINDOW_AUTOSIZE)
# cv2.setMouseCallback("VideoFrame", captuer)
#
# while True:
#     ret, frame = capture.read()
#     # x=80 ,y =87
#     # x2=465 , y2=345
#     line = frame[87:300, 80:465]  #[y:y2,x:x2]
#     gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
#
#     th, src_bin = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
#     src_bin = cv2.bitwise_not(src_bin) #흑백 반전
#
#     cv2.namedWindow('line', cv2.WINDOW_AUTOSIZE)
#     cv2.imshow("line", src_bin)
#
#     cv2.namedWindow('VideoFrame', cv2.WINDOW_AUTOSIZE)
#     cv2.imshow("VideoFrame", frame)
#     key = cv2.waitKey(1)
#
#     if key & 0xFF == ord('q') or key == 27:
#         cv2.destroyAllWindows()
#         break

img_new=cv2.imread('linepoint.jpg')
plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
plt.show()

#############################

cv2.waitKey()
cv2.destroyAllWindows()

# line 좌표 설정

img = cv2.imread("linepoint.jpg")
# src = cv2.resize(img,(1024,1024))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th, src_bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
# src_bin = cv2.bitwise_not(src_bin)

contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

plt.imshow(cv2.cvtColor(src_bin, cv2.COLOR_BGR2RGB))
plt.show()

i=0
points1=[]

for pts in contours:
    # 너무 작은 객체는 제외
    if cv2.contourArea(pts) < 1000:
        continue
    # 외곽선 근사화
    image = cv2.drawContours(img, [pts], -1, (0, 0, 0), 1)

    ## 중심 구하기
    rect = cv2.minAreaRect(pts)
    (x_center, y_center), (w, h), angel = rect

    # x,y 좌표 추출
    for k in pts:
        x,y = k.ravel()
        # 포인트 리스트 저장
        points1.append({'x':x,'y': y})

# 포인트 x 기준으로 정렬
sort_points1=sorted(points1,key=(lambda x:x['x']))
# sort_points1=sorted(points1,key=(lambda x:x['y']))

# 중복 x 제거
sort_points_x=list({point['x']:point for point in sort_points1}.values())
# sort_points_x=list({point['y']:point for point in sort_points1}.values())
cnt=0



for i in range(len(sort_points_x)):
    sort_points_x[i]['idx']=cnt
    cnt += 1

sort_points_x[0]['angel'] = int(np.degrees(np.arctan((y_center-sort_points_x[0]['y']) / (x_center-sort_points_x[0]['x']))))
points2=[sort_points_x[0]]

# print('sort_points_x:',sort_points_x)
# print(len(sort_points_x))

print(points2)
i=0
# 젠가 높이 고려
k=110
# 젠가 높이를 고려해서 라인 좌표 구하기
for d1 in sort_points_x:
    if d1['idx'] == points2[i]['idx']:
        for d2 in sort_points_x:
            if d1['idx'] >= d2['idx']:
                continue
            distance = math.sqrt(math.pow(d1['x'] - d2['x'], 2) + math.pow(d1['y'] - d2['y'], 2))
            if distance<(k*0.9) and distance>(k*0.3):
                
                ### 기울기를 이용해 각도 구하기, 원점에서 좌표 tan각도 이용
                dx = abs(x_center - d2['x'])
                dy = abs(y_center - d2['y'])

                angeldff = int(np.degrees(np.arctan(dy / dx)))
                print(dx, dy,angeldff)
                points2.append({'x':d2['x'],'y':d2['y'],'angel':angeldff,'idx':d2['idx']})
                i+=1
                break

#####################################

#################################

point_final=[]
f=open("point_final.txt",'w')
for i in range(0,15):
    x=points2[i]['x']
    y=points2[i]['y']
    if i ==0:
        angel=points2[i]['angel'] ##초기 각도
        point_final.append({'x': x, 'y': y, 'angel': angel, 'idx': i}) ## 현재 각도에서 그 전 각도 빼서 각도 변화량 추출

        continue
    angel = points2[i]['angel']-points2[i-1]['angel']
    point_final.append({'x': x, 'y': y, 'angel': angel, 'idx': i})

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(x_center, y_center, c="y", s=30)
    plt.scatter(x, y, c="r", s=30)
    # plt.text(x, y, f'({x},{y})',bbox={'boxstyle':'round','ec':(1.0,0.5,0.5),'fc':(1.0,0.8,0.8)})
    plt.text(x + 5, y + 20, f'({angel}°)', bbox={'boxstyle': 'round', 'ec': (1.0, 0.5, 0.5), 'fc': (1.0, 0.8, 0.8)})

for i in range(len(point_final)):
    x = point_final[i]['x']
    y = point_final[i]['y']
    angel = point_final[i]['angel']
    # k= "["+str(x)+","+ str(y)+","+ str(angel)+"]"
    k = str(x) + ' ' + str(y) + ' ' + str(angel)+' '
    with open('point_final.txt', "a") as f:
        f.write(f'{k}')
    # 메시지를 전송합니다.



plt.show()
print("point_final",point_final)

###txt 파일 읽기
f = open("point_final.txt", 'r')
line = f.readline()

print(line)
client_socket.sendall(line.encode())
f.close()

# 메시지를 수신합니다.
data = client_socket.recv(1024)
print('Received', repr(data.decode()))

# 소켓을 닫습니다.
client_socket.close()