# coding=utf-8
import streamlit as st
from PIL import Image
import cv2 as cv
import tempfile
import numpy as np

st.write("""
# 生成图片中图像的轮廓
控制阈值寻找图片中图像的轮廓, 针对病毒检测的膜条图像, 进行膜条图像获取, 并标记膜条的位置
""")

col1, col2 = st.columns(2)

image = Image.open('samples/src.jpg')
with col1:
    st.image(image)

contours = Image.open('samples/dst.png')
with col2:
    st.image(contours)

inSrc = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg', 'bmp'])

if inSrc is None:
    src = cv.imread('samples/src.jpg')
else:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(inSrc.read())

    src = cv.imread(tfile.name)

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

thresh = st.slider("二值化阈值", 0, 255, 108, step=1)
maxval = st.slider("二值化最大值", 0, 255, 200, step=1)

# 二值化
ret, binary = cv.threshold(gray, thresh, maxval, cv.THRESH_BINARY)

erode_kernel_sizex = st.slider("腐蚀核尺寸 (x)", 3, 255, 3, step=1)
erode_kernel_sizey = st.slider("腐蚀核尺寸 (y)", 3, 255, 30, step=1)

# 腐蚀
kernel = cv.getStructuringElement(cv.MORPH_RECT, (erode_kernel_sizex, erode_kernel_sizey))
erode = cv.erode(binary, kernel, iterations=1)

dilate_kernel_sizex = st.slider("膨胀核尺寸 (x)", 3, 255, 210, step=1)
dilate_kernel_sizey = st.slider("膨胀核尺寸 (y)", 3, 255, 80, step=1)

# 膨胀
kernel = cv.getStructuringElement(cv.MORPH_RECT, (dilate_kernel_sizex, dilate_kernel_sizey))
dilate = cv.dilate(erode, kernel, iterations=1)

# 查找轮廓
dst, h = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

ouCol1, ouCol2 = st.columns(2)

black = np.zeros(src.shape)
output = cv.drawContours(black, dst, -1, (0,225,0), 3)
forOutput = output.astype("uint8")

toImage = Image.fromarray(cv.cvtColor(forOutput, cv.COLOR_BGR2RGB))

with ouCol1:
    st.image(toImage)

srcOutput = cv.drawContours(src, dst, -1, (0,225,0), 3)

toSrcImage = Image.fromarray(cv.cvtColor(srcOutput, cv.COLOR_BGR2RGB))
#cv.imwrite("samples/new.png", srcOutput)
with ouCol2:
    st.image(toSrcImage)