# AI-Image-Analysis-Pipeline
 An AI Image Analysis Pipeline automates image processing by detecting objects, segmenting them, extracting features, and generating textual descriptions or structured data.
*****


https://github.com/user-attachments/assets/2f57f3ae-50b9-4f8c-aea1-c2e4bcff63ed


## Installation
1. Clone the Repository
``` bash
git clone https://github.com/Rishi-Jain2602/AI-Image-Analysis-Pipeline.git
```
2. Create Virtual Environment
```bash
virtualenv venv
venv\Scripts\activate
```
3. Install the Project dependencies
```bash
pip install -r requirements.txt
```
4. To Run Streamlit app
```bash
cd streamlit_app
streamlit run app.py
```

****

This Jupyter Notebook link: https://github.com/Rishi-Jain2602/Rishi-Jain/blob/main/tables/models/wasserstoff-ai-intern.ipynb

This will provide detailed explanations for each step I took. If the Jupyter notebook doesn't work locally, try running it on Kaggle.
****
### Tools & Models Used:

**1. Image Segmentation:** Mask R-CNN

**2. Feature Extraction:** YOLO (Hugging Face Model: hustvl/yolos-tiny)

**3. Text/Data Extraction from Objects:** Hugging Face Model (Salesforce/blip-image-captioning-large)

**4. Summarization:** Hugging Face Model (facebook/bart-large-cnn)


### Pipeline Structure:

**1. Input:** Upload an image.

**2. Processing:** Detect, segment, extract features, and generate captions.

**3. Output:** Annotated image and summary table.

**Integration:** Streamlit app for seamless user interaction and review.

****

## Result

![annotated_image](https://github.com/user-attachments/assets/3079d0bc-3f65-4958-9e9c-e7bc757288ad)



## Note
1. Make sure you have Python 3.x installed
2. It is recommended to use a virtual environment to avoid conflict with other projects.
3. If the Jupyter notebook doesn't work locally, try running it on Kaggle.
4. For deep learning, a laptop with a powerful GPU, a high-performance CPU, at least 8GB of RAM, a fast SSD, and an efficient cooling system is recommended.
5. If you encounter any issue during installation or usage please contact rishijainai262003@gmail.com or rj1016743@gmail.com




