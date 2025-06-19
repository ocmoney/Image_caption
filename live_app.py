import streamlit as st
from safetensors.torch import load_file
from model import CaptionGenerator, get_tokenizer
from evaluate import predict_caption
from PIL import Image
import cv2
import numpy as np
from transformers import AutoImageProcessor
import time as time_module

st.title("Live Image Captioner")

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

model = load_model(num_heads=6, num_layers=8, _tokenizer=load_tokenizer(), run_name="celestial-serenity-1", epoch=1)
tokenizer = load_tokenizer()
image_processor = load_image_processor()

st.write("Live video captioning - Continuous caption feed from camera")

# Initialize session state for live feed
if 'live_feed_active' not in st.session_state:
    st.session_state.live_feed_active = False
if 'last_caption_time' not in st.session_state:
    st.session_state.last_caption_time = 0
if 'captions_history' not in st.session_state:
    st.session_state.captions_history = []
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_processed_frame' not in st.session_state:
    st.session_state.last_processed_frame = None

# Add controls for live processing
col1, col2, col3 = st.columns(3)
with col1:
    live_mode = st.checkbox("Enable live video feed", value=st.session_state.live_feed_active)
with col2:
    if live_mode:
        live_interval = st.slider("Processing interval (seconds)", 1, 10, 3)
with col3:
    if live_mode:
        max_history = st.slider("Max captions to show", 1, 10, 5)

# Update session state
st.session_state.live_feed_active = live_mode

if live_mode:
    st.write("📹 Live video feed mode - Automatically capturing and processing frames")
    
    # Add controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Stop Live Feed"):
            st.session_state.live_feed_active = False
            st.rerun()
    with col2:
        if st.button("Clear History"):
            st.session_state.captions_history = []
            st.rerun()
    with col3:
        if st.button("Manual Capture"):
            st.session_state.manual_capture = True
            st.rerun()
    
    # Live feed camera input with dynamic key to force new captures
    camera_key = f"live_feed_camera_{st.session_state.frame_count}"
    camera_photo = st.camera_input("Live Video Feed", key=camera_key)
    
    # Check if camera is active - only use current photo, not stored ones
    camera_active = camera_photo is not None
    
    if camera_active:
        # Convert the photo to PIL Image
        image = Image.open(camera_photo)
        
        # Display the current frame
        st.image(image, caption=f'Frame {st.session_state.frame_count}', use_column_width=True)
        
        # Check if we should process this frame based on interval
        current_time = time_module.time()
        should_process = (
            current_time - st.session_state.last_caption_time >= live_interval or
            st.session_state.get('manual_capture', False) or
            st.session_state.frame_count == 0  # Always process first frame
        )
        
        if should_process:
            st.write("🔄 Processing frame automatically...")
            
            # Process the image
            image_pixels = image_processor(image, return_tensors="pt")["pixel_values"]
            
            # Generate caption
            start_time = time_module.time()
            caption_text = predict_caption(model, image_pixels, tokenizer)
            processing_time = time_module.time() - start_time
            
            # Add to history
            caption_entry = {
                'text': caption_text,
                'time': time_module.strftime('%H:%M:%S'),
                'processing_time': processing_time,
                'frame': st.session_state.frame_count
            }
            st.session_state.captions_history.append(caption_entry)
            
            # Keep only recent captions
            if len(st.session_state.captions_history) > max_history:
                st.session_state.captions_history = st.session_state.captions_history[-max_history:]
            
            # Update last caption time and frame count
            st.session_state.last_caption_time = current_time
            st.session_state.manual_capture = False
            st.session_state.frame_count += 1
            
            # Show processing status
            st.success(f"✅ Auto-caption generated in {processing_time:.2f}s")
        
        # Display caption history
        if st.session_state.captions_history:
            # Create a continuous paragraph from recent captions
            recent_captions = st.session_state.captions_history[-3:]  # Get last 3 captions
            caption_words = []
            
            for entry in recent_captions:
                words = entry['text'].split()
                caption_words.extend(words)
            
            # Keep only the last 20 words
            if len(caption_words) > 20:
                caption_words = caption_words[-20:]
            
            # Create the continuous paragraph
            continuous_caption = " ".join(caption_words)
            
            st.write("### Live Caption Feed:")
            st.write(f"**{continuous_caption}**")
            
            # Show processing info
            latest_entry = st.session_state.captions_history[-1]
            st.caption(f"Last updated: {latest_entry['time']} | Processing time: {latest_entry['processing_time']:.2f}s")
        
        # Auto-refresh for continuous live feed
        if live_mode:
            # Use JavaScript to auto-trigger camera capture
            st.markdown(
                f"""
                <script>
                    // Auto-trigger camera capture after a delay
                    setTimeout(function() {{
                        // Find and click the camera capture button
                        const buttons = document.querySelectorAll('button');
                        for (let button of buttons) {{
                            if (button.textContent.includes('Take photo') || button.textContent.includes('📷')) {{
                                button.click();
                                break;
                            }}
                        }}
                    }}, 1000);
                    
                    // Auto-refresh the page after processing
                    setTimeout(function() {{
                        window.location.reload();
                    }}, {live_interval * 1000});
                </script>
                """,
                unsafe_allow_html=True
            )
            st.info(f"🔄 Auto-capturing and refreshing every {live_interval} seconds...")
            
            # Add a progress bar to show countdown
            progress_bar = st.progress(0)
            for i in range(live_interval):
                time_module.sleep(1)
                progress_bar.progress((i + 1) / live_interval)
            
            st.rerun()
    
    else:
        st.info("📹 Click 'Take Photo' in the camera above to start live feed processing")
        
    # Show live feed status
    if st.session_state.live_feed_active:
        if camera_active:
            st.success(f"🔄 Live feed active - Auto-capturing every {live_interval} seconds")
        else:
            st.info(f"🔄 Live feed active - Click 'Take Photo' to start")
    else:
        st.warning("⏸️ Live feed stopped")
else:
    st.info("📹 Enable live video feed to start continuous captioning")
