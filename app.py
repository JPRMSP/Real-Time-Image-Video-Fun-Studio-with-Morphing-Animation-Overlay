import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageSequence
import tempfile
import imageio

st.set_page_config(page_title="🎨 Real-Time Image & Video Fun Studio", layout="wide")
st.title("🎨 Real-Time Image & Video Fun Studio with Morphing & Animation Overlay")

option = st.sidebar.selectbox("Choose Mode", ["Image Processing", "Image Morphing", "Video Processing"])

# --- Filters ---
def apply_filters(img, filter_type):
    if filter_type == "Grayscale":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif filter_type == "Low Pass Filter":
        kernel = np.ones((5,5),np.float32)/25
        return cv2.filter2D(img,-1,kernel)
    elif filter_type == "High Pass Filter":
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        return cv2.filter2D(img,-1,kernel)
    elif filter_type == "Laplacian":
        return cv2.Laplacian(img, cv2.CV_64F)
    elif filter_type == "Edge Detection":
        return cv2.Canny(img,100,200)
    elif filter_type == "Histogram Stretch":
        min_val = np.min(img)
        max_val = np.max(img)
        stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return stretched
    else:
        return img

# --- Image Processing ---
if option == "Image Processing":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])
    filter_type = st.selectbox("Select Filter/Operation", ["Original", "Grayscale", "Low Pass Filter", "High Pass Filter", "Laplacian", "Edge Detection", "Histogram Stretch"])
    overlay_file = st.file_uploader("Upload Animation/GIF Overlay (Optional)", type=["gif","png","jpg"])
    overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5)
    if uploaded_file:
        image = np.array(Image.open(uploaded_file).convert('RGB'))
        processed = apply_filters(image, filter_type)
        # Overlay
        if overlay_file:
            overlay = Image.open(overlay_file).convert('RGBA').resize((processed.shape[1], processed.shape[0]))
            processed = Image.alpha_composite(Image.fromarray(processed).convert('RGBA'), overlay.putalpha(int(overlay_alpha*255)))
        st.image(processed, caption="Processed Image", use_column_width=True)

# --- Image Morphing ---
elif option == "Image Morphing":
    st.info("Upload two images to morph between them")
    img1_file = st.file_uploader("Upload Image 1", type=["jpg","png","jpeg"], key="img1")
    img2_file = st.file_uploader("Upload Image 2", type=["jpg","png","jpeg"], key="img2")
    morph_factor = st.slider("Morph Factor", 0.0, 1.0, 0.5)
    if img1_file and img2_file:
        img1 = np.array(Image.open(img1_file).convert('RGB'))
        img2 = np.array(Image.open(img2_file).convert('RGB'))
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        morphed = cv2.addWeighted(img1, 1-morph_factor, img2_resized, morph_factor, 0)
        st.image(morphed, caption=f"Morphed Image (Factor={morph_factor})", use_column_width=True)

# --- Video Processing ---
elif option == "Video Processing":
    video_file = st.file_uploader("Upload a Video", type=["mp4","avi","mov"])
    filter_type = st.selectbox("Select Filter/Operation", ["Original", "Grayscale", "Low Pass Filter", "High Pass Filter", "Laplacian", "Edge Detection"])
    overlay_file = st.file_uploader("Upload Animation/GIF Overlay (Optional)", type=["gif","png","jpg"])
    overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5)
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        vidcap = cv2.VideoCapture(tfile.name)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frames = []
        # GIF overlay
        gif_frames = None
        if overlay_file:
            overlay_gif = Image.open(overlay_file).convert('RGBA')
            gif_frames = [np.array(frame.convert('RGBA').resize((int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))))) for frame in ImageSequence.Iterator(overlay_gif)]
        idx = 0
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = apply_filters(frame, filter_type)
            if len(processed_frame.shape)==2:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
            # Overlay
            if gif_frames:
                overlay_frame = gif_frames[idx % len(gif_frames)]
                alpha = overlay_alpha
                processed_frame = cv2.addWeighted(processed_frame, 1-alpha, overlay_frame[:,:,:3], alpha, 0)
                idx += 1
            frames.append(processed_frame)
        temp_output = "processed_video.mp4"
        imageio.mimsave(temp_output, frames, fps=fps)
        st.video(temp_output)
        st.download_button("Download Video", data=open(temp_output,"rb"), file_name="processed_video.mp4", mime="video/mp4")
