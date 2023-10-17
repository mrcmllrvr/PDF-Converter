# Python Built-in Libraries
import os
import json
import uuid
from io import BytesIO
import cv2
import numpy as np
import multiprocessing
import time

# Third-party Libraries
import requests
import streamlit as st
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from pytesseract import image_to_string, pytesseract
from dotenv import load_dotenv
import pypdfium2 as pdfium

# Local Libraries
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import tempfile
import shutil


# Load environment variables
load_dotenv()



files_to_delete = []

def delayed_delete():
    """
    Delayed file deletion mechanism.
    """
    for file in files_to_delete:
        safe_delete(file)

def safe_delete(filename, retries=5, delay=5):
    """Attempt to delete a file with multiple retries and a delay."""
    for _ in range(retries):
        try:
            if os.path.isdir(filename):
                shutil.rmtree(filename)
            else:
                os.remove(filename)
            return
        except PermissionError as e:
            st.text(f"Attempt {_ + 1} to delete {filename} failed due to: {str(e)}")
            time.sleep(delay)
    raise PermissionError(f"Could not delete file {filename} after {retries} attempts.")


# Modify create_temp_file to use a temporary directory
def create_temp_file() -> str:
    temp_dir = tempfile.mkdtemp()
    files_to_delete.append(temp_dir)  # Add this to ensure we clean up later
    return os.path.join(temp_dir, "tempfile.pdf")

# # 1. Convert PDF file into images via pypdfium2
#@st.cache
def convert_pdf_to_images(file_path, scale=500/72):
    st.text(f"Processing PDF at path: {file_path}")
    pdf_file = pdfium.PdfDocument(file_path)

    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    final_images = []
    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))
        image_byte_array.close()

    # Close the pdf_file, if the library has a close or similar method
    if hasattr(pdf_file, "close"):
        pdf_file.close()

    return final_images

def preprocess_image(img, threshold=200):
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)
    img = img.filter(ImageFilter.MedianFilter())
    
    # Use adaptive thresholding
    img_np = np.array(img)
    adaptive_thresh = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = Image.fromarray(adaptive_thresh)
    
    # Binarization with threshold
    img = img.point(lambda x: 0 if x < threshold else 255, '1')
    return img


# 2. Extract text from images via pytesseract
#@st.cache
def extract_text_from_img(list_dict_final_images):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):
        image = Image.open(BytesIO(image_bytes))
        st.image(image, caption=f'Page {index+1}', use_column_width=True)  # Displaying the image to Streamlit
        image.close()

        image_temp_path = f"temp_img_{uuid.uuid4()}.jpeg"
        image.save(image_temp_path)
        
        raw_text = enhanced_ocr(image_temp_path)  # Using enhanced_ocr for better table extraction
        st.text(raw_text)  # Display extracted text to Streamlit
        
        os.remove(image_temp_path)
        image_content.append(raw_text)

    return "\n".join(image_content)

def enhanced_ocr(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1,1), np.uint8)
    dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)
    pil_img = Image.fromarray(dilated_img)
    text = pytesseract.image_to_string(pil_img, config='--oem 3 --psm6')
    return text

#@st.cache
def extract_content_from_file_path(file_path: str):
    images_list = convert_pdf_to_images(file_path)
    text_with_pytesseract = extract_text_from_img(images_list)
    return text_with_pytesseract


# 3. Extract structured info from text via LLM
def extract_structured_data(content: str, data_points):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    template = """
    You are an expert admin people who will extract core information from documents

    {content}

    Above is the content; please try to extract all data points from the content above 
    and export in a JSON array format:
    {data_points}

    Now please extract details from the content  and export in a JSON array format, 
    return ONLY the JSON array:
    """

    prompt = PromptTemplate(
        input_variables=["content", "data_points"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    results = chain.run(content=content, data_points=data_points)
    return results

def process_file(args):
    file, data_points  = args
    temp_filename = create_temp_file()
    filename = file.name
    with open(temp_filename, 'wb') as f:
        f.write(file.getbuffer())
    try:
        content = extract_content_from_file_path(temp_filename)
        data = extract_structured_data(content, data_points)
        json_data = json.loads(data)
        for item in json_data:
            item['filename'] = filename
        file_results = json_data if isinstance(json_data, list) else [json_data]
    except Exception as e:
        st.error(f"Error processing file {filename}: {e}")
        file_results = []
    
    #safe_delete(temp_filename)

    return file_results


# TODO: Uncomment and refine the following function if needed

# 4. Send data to make.com via webhook
# def send_to_make(data):
#     webhook_url = "https://hook.eu1.make.com/xxxxxxxxxxxxxxxxx"
#     json_data = {"data": data}
#     try:
#         response = requests.post(webhook_url, json=json_data)
#         response.raise_for_status()  
#         print("Data sent successfully!")
#     except requests.exceptions.RequestException as e:
#         print(f"Failed to send data: {e}")


# 5. Streamlit app
def main():
    # default_data_points = {
    #     "Name": "name of the employee",
    #     "Net Pay": "how much does the employee earn",
    #     "Company Name": "company that the employee is working for",
    #     "Basic Pay": "Basic pay of the employee"
    # }
    # default_data_points_str = json.dumps(default_data_points, indent=4)

    st.set_page_config(page_title="PDF Extraction", page_icon=":bird:")
    st.header("PDF Extraction :bird:")

    # data_points = st.text_area("Data points", value=default_data_points_str, height=170)
    # uploaded_files = st.file_uploader("upload PDFs", accept_multiple_files=True, type=["pdf"])

    # threshold = st.slider('Set OCR Binarization Threshold', 50, 200, 140)

    # results = []

    # if uploaded_files:
    #     with multiprocessing.Pool() as pool:
    #         results = pool.map(process_file, [(file, data_points) for file in uploaded_files])
    #     results = [item for sublist in results for item in sublist]  # flatten list of lists

    #     if results:
    #         df = pd.DataFrame(results)
    #         st.subheader("Results")
    #         st.data_editor(df)
    #         df.to_excel('extracted_data.xlsx', index=False)
    #         st.write("Data saved to extracted_data.xlsx!")

    # # Add a button to manually trigger the cleanup
    # if st.button("Cleanup Temporary Files"):
    #     delayed_delete()

    # # Add a button to manually shutdown the Streamlit server
    # if st.button("Shutdown Streamlit Server"):
    #     os._exit(0)


    if __name__ == '__main__':
    # multiprocessing.freeze_support()
        main()

