from PIL import Image
import io
import requests

import fitz  # PyMuPDF

def image_to_proper_pdf(image_path, pdf_path):
    doc = fitz.open()
    img = fitz.Pixmap(image_path)
    rect = fitz.Rect(0, 0, img.width, img.height)

    page = doc.new_page(width=img.width, height=img.height)
    page.insert_image(rect, pixmap=img)

    doc.save(pdf_path)
    doc.close()

pdf_path = "ocr.pdf"
image_to_proper_pdf("ocr.png", pdf_path)

# Huridocs에 보내기
url_ocr = "http://localhost:8051/ocr"
url = "http://localhost:8051"
files = {'file': ('ocr.pdf', open(pdf_path, 'rb'), 'application/pdf')}

response = requests.post(url_ocr, files=files)
print(response.status_code)
#print(response.text)

output_pdf_path = "ocr_result.pdf"
with open(output_pdf_path, "wb") as f:
    f.write(response.content)

with open("ocr_result.pdf", "rb") as f:
    response = requests.post(
        url,
        files={"file": ("ocr_result.pdf", f, "application/pdf")}
    )

layout_result = response.json()
print(layout_result)