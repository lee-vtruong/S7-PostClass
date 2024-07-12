from PIL import Image
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import requests

def segment_image(img_pil, k):
    img = np.array(img_pil)
    X = img.reshape(img.shape[0] * img.shape[1], 4)

    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(X)

    img_new = kmeans.cluster_centers_[kmeans.labels_]
    img_new = img_new.reshape(img.shape[0], img.shape[1], 4)
    img_new = img_new.astype(np.uint8)
    return Image.fromarray(img_new)

def app():
    st.set_page_config(page_title='Le Van Truong - S7 Postclass')
    st.title("Image Segmentation with kMeans")
    left_col, right_col = st.columns(2)
    image_url = left_col.text_input("Image URL (Press Enter to apply)")
    k_value = right_col.slider("K", 2, 10, 5)

    if image_url:
        image = Image.open(requests.get(image_url, stream=True).raw)
        left_col, right_col = st.columns(2)
        left_col.write("Original Image")
        left_col.image(image)

        segmented_image = segment_image(image.copy(), k_value)
        right_col.write("Segmented Image")
        right_col.image(segmented_image)

if __name__ == "__main__":
    app()
