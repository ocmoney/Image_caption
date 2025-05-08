import streamlit as st
from safetensors.torch import load_file
from model import CaptionGenerator, get_tokenizer
from evaluate import predict_caption
from PIL import Image
from transformers import AutoImageProcessor
from time import time

st.title("Image Captioner")

@st.cache_resource
def load_model(num_heads, num_layers, _tokenizer, run_name, epoch):
    model = CaptionGenerator(num_heads=num_heads, num_layers=num_layers, tokenizer=_tokenizer)
    model.load_state_dict(load_file(f"model/{run_name}/transformer_{epoch-1}.safetensors"))
    return model

@st.cache_resource
def load_tokenizer():
    return get_tokenizer()

@st.cache_resource
def load_image_processor():
    return AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

model = load_model(num_heads=6, num_layers=8, _tokenizer=load_tokenizer(), run_name="cosmic-yogurt-52", epoch=10)
tokenizer = load_tokenizer()
image_processor = load_image_processor()

st.write("Upload an image to generate a caption")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_pixels = image_processor(image, return_tensors="pt")["pixel_values"]
    st.image(image, caption='Uploaded Image', use_column_width=True)

    start_time = time()
    text = predict_caption(model, image_pixels, tokenizer)
    st.write(text)
    st.write(f"Time taken: {time() - start_time} seconds")
