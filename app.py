import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw
import pandas as pd
import io

# Define functions for each step
def segment_and_identify_objects(image):
    pipe = pipeline("object-detection", model="hustvl/yolos-tiny")
    results = pipe(image)
    return results

def extract_text_from_image(image):
    pipe_img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text = pipe_img_to_text(image)
    return text[0]['generated_text'] if text else ""

def annotate_image(original_image, object_positions):
    draw = ImageDraw.Draw(original_image)
    for index, (x1, y1, x2, y2) in enumerate(object_positions):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"Object {index+1}", fill="red")
    return original_image

# Streamlit app
st.title("AI Image Analysis Pipeline")

# Step 1: Upload an Image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 2 & 3: Segment and Identify Objects
    st.write("Segmenting and Identifying Objects...")
    results = segment_and_identify_objects(image)
    
    object_positions = []
    extracted_texts = []
    
    for i, result in enumerate(results):
        box = result["box"]
        object_positions.append((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
        
        # Crop the object from the image
        object_image = image.crop((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
        
        # Step 4: Image to Text using Hugging Face Pipeline
        text = extract_text_from_image(object_image)
        extracted_texts.append(text)
        
        st.write(f"Object {i+1}:")
        st.image(object_image, caption=f"Object {i+1}")
        st.write(f"Extracted Text: {text}")
    
    # Step 5 & 6: Data Mapping and Output Generation
    mapped_data = []
    for i in range(len(object_positions)):
        mapped_data.append({
            "object_id": f"object_{i+1}",
            "position": object_positions[i],
            "extracted_text": extracted_texts[i],
        })
    
    st.write("Generating Annotated Image and Summary Table...")
    
    annotated_image = annotate_image(image.copy(), object_positions)
    
    # Save annotated image to a BytesIO object for download
    img_bytes = io.BytesIO()
    annotated_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    st.image(annotated_image, caption="Annotated Image with Objects", use_column_width=True)
    
    # Display Summary Table
    df = pd.DataFrame(mapped_data)
    st.write("Summary Table:")
    st.table(df)
    
    # Provide Download Links
    st.download_button("Download Annotated Image", data=img_bytes, file_name="annotated_image.png", mime="image/png")
    st.download_button("Download Summary Table CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="summary_table.csv", mime="text/csv")
