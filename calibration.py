# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:54:56 2021

@author: 82106
"""

# 로봇으로 인식 된 물체로 이동 시켜 로봇 위치 좌표를 측정 한다.
# 물체 3개에 대한 좌표를 추출한 후에 dx,dy에 넣는다.
#이미지 행렬 [x,y,1]
import numpy as np

def calibration_jengar(x,y):
    ### dx 행렬
    X=np.mat([[-0.252997918],
          [0.329915104],
          [0]])


    ### dy 행렬
    Y=np.mat([[0.021884545],
          [-0.014027422],
          [0]])


    ### 이미지 행렬
    Prime=np.mat([[41, 273, 1],
              [596, 271, 1],
              [324, 270, 1]])


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


    r_x =-0.000761719


    r_y = 0.42677779


    ### Calibration 식
    dx = dx_matrix[0, 0]*image_x + dx_matrix[1, 0]*image_y + dx_matrix[2, 0]
    dy = dy_matrix[0, 0]*image_x + dy_matrix[1, 0]*image_y + dy_matrix[2, 0]



    print(r_x + dx)
    print(r_y + dy)

calibration_jengar(10,11)

def calibration_line(x, y):
    ### dx 행렬
    X = np.mat([[-0.252997918],
                [0.329915104],
                [0]])

    ### dy 행렬
    Y = np.mat([[0.021884545],
                [-0.014027422],
                [0]])

    ### 이미지 행렬
    Prime = np.mat([[41, 273, 1],
                    [596, 271, 1],
                    [324, 270, 1]])

    ### dx, dy 행렬
    dx_matrix = Prime.I * X
    dy_matrix = Prime.I * Y

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

    r_x = -0.000761719

    r_y = 0.42677779

    ### Calibration 식
    dx = dx_matrix[0, 0] * image_x + dx_matrix[1, 0] * image_y + dx_matrix[2, 0]
    dy = dy_matrix[0, 0] * image_x + dy_matrix[1, 0] * image_y + dy_matrix[2, 0]

    print(r_x + dx)
    print(r_y + dy)


