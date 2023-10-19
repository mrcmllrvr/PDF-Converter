import uuid
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from pytesseract import image_to_string
from PIL import Image, ImageEnhance
from io import BytesIO
import pypdfium2 as pdfium
import streamlit as st
import multiprocessing
import pandas as pd
import json
import os
import requests
import io

load_dotenv()

# 1. Convert PDF file into images via pypdfium2
def convert_pdf_to_images(file_path, scale=500/72):
    print(f"Processing PDF at path: {file_path}")
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

    # Close the pdf_file, if the library has a close or similar method
    if hasattr(pdf_file, "close"):
        pdf_file.close()

    return final_images

def preprocess_image(img):
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    return img  # Only return the preprocessed image

# 2. Extract text from images via pytesseract
def extract_text_from_img(list_dict_final_images):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):
        image = Image.open(io.BytesIO(image_bytes))
        
        # Display the image using st.image
        st.image(image, caption=f'Page {index + 1}', use_column_width=True)
        
        processed_image = preprocess_image(image)
        raw_text = str(image_to_string(processed_image, config='--oem 3 --psm 6'))

        # Display the raw OCR content to check for accuracy
        st.text(raw_text)

    image_content.append(raw_text)

    return "\n".join(image_content)

def extract_content_from_url(url: str):
    images_list = convert_pdf_to_images(url)
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
    default_data_points = """{
        "Name": "name of the employee",
        "Net Pay": "how much does the employee earn",
        "Company Name": "company that the employee is working for",
        "Basic Pay": "Basic pay of the employee"
    }"""

    st.set_page_config(page_title="PDF Extraction", page_icon=":bird:")
    st.header("PDF Extraction :bird:")

    data_points = st.text_area("Data points", value=default_data_points, height=170)
    uploaded_files = st.file_uploader("upload PDFs", accept_multiple_files=True)

    if uploaded_files is not None and data_points is not None:
        results = []
        for file in uploaded_files:
            temp_filename = f"temp_{uuid.uuid4()}.pdf"
            filename = file.name  
            with open(temp_filename, 'wb') as f:
                f.write(file.getbuffer())

            try:
                content = extract_content_from_url(temp_filename)
                data = extract_structured_data(content, data_points)
                json_data = json.loads(data)

                for item in json_data:
                    item['filename'] = filename

                results.extend(json_data) if isinstance(json_data, list) else results.append(json_data)
            except Exception as e:
                st.error(f"Error processing file {filename}: {e}")

            # Removing the temporary file
            try:
                os.remove(temp_filename)
            except PermissionError:
                print(f"Warning: Could not delete temporary file {temp_filename}.")

        if results:
            df = pd.DataFrame(results)
            st.subheader("Results")
            st.dataframe(df)
            
            # Save DataFrame to a BytesIO object
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            
            # Create a download button for the Excel file
            st.download_button(
                label="Download extracted data",
                data=output,
                file_name='extracted_data.xlsx',
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
