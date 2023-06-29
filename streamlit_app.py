# coding=utf-8
import streamlit as st
from PIL import Image
import cv2 as cv
import tempfile
import numpy as np

st.write("""
# 生成图片中图像的轮廓
控制阈值寻找图片中图像的轮廓
""")

col1, col2 = st.columns(2)

image = Image.open('samples/src.jpg')
with col1:
    st.image(image)

contours = Image.open('samples/src.jpg')
with col2:
    st.image(contours)

threthold = st.slider("阈值", 0, 200, step=1)

src = cv.imread('samples/src.jpg')
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# 二值化
ret, binary = cv.threshold(gray, 150, 200, cv.THRESH_BINARY)

# 腐蚀
kernel = np.ones((25, 25), int)
erode = cv.erode(binary, kernel, iterations=1)

# 膨胀
kernel = np.ones((10, 10), int)
dilate = cv.dilate(erode, kernel, iterations=1)

# 查找轮廓
dst, h = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

black = np.zeros(src.shape)
output = cv.drawContours(black, dst, -1, (0,225,0), 3)

toImage = Image.fromarray(cv.cvtColor(output, cv.COLOR_BGR2RGB))
st.image(toImage)