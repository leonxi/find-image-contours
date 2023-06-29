# coding=utf-8
import streamlit as st
from PIL import Image
import cv2 as cv
import tempfile

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
ret, binary = cv.threshold(gray, 150, 200, cv.THRESH_BINARY)
dst, h = cv.findContours(src, cv.RETR_FLOODFILL, cv.CHAIN_APPROX_SIMPLE)

toImage = Image.fromarray(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
st.image(toImage)