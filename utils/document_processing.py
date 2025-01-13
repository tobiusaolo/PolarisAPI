import pdfplumber 
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
import pandas as pd
import pytesseract
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException
from io import BytesIO, StringIO
import fitz 
from zipfile import ZipFile, BadZipFile

def validate_docx(content: bytes):
    try:
        with ZipFile(BytesIO(content)) as docx_zip:
            if "word/document.xml" not in docx_zip.namelist():
                raise HTTPException(status_code=400, detail="Invalid DOCX file structure.")
    except BadZipFile:
        raise HTTPException(status_code=400, detail="Corrupted DOCX file.")
    

def parse_document(content: bytes, filename: str):
    """
    Parse document content into text chunks based on file type.

    Args:
        content (bytes): The binary content of the uploaded file.
        filename (str): The name of the uploaded file.

    Returns:
        List[str]: A list of text chunks extracted from the document.
    """
    try:
        if filename.endswith('.pdf'):
            # Primary PDF parser using pdfplumber
            text_chunks = []
            try:
                with pdfplumber.open(BytesIO(content)) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_chunks.append(text)
                        else:
                            # Fallback to OCR for scanned or blank pages
                            image = page.to_image().original
                            text_chunks.append(pytesseract.image_to_string(Image.open(image)))
            except Exception as e:
                # Fallback to PyMuPDF if pdfplumber fails
                doc = fitz.open(stream=BytesIO(content), filetype="pdf")
                for page in doc:
                    text = page.get_text()
                    if text.strip():
                        text_chunks.append(text)
                    else:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text_chunks.append(pytesseract.image_to_string(img))
            return text_chunks

        elif filename.endswith('.docx'):
            # Validate DOCX integrity
            validate_docx(content)

            # Parse DOCX using python-docx
            text_chunks = []
            try:
                with BytesIO(content) as doc_stream:
                    doc = Document(doc_stream)
                    for paragraph in doc.paragraphs:
                        if paragraph.text.strip():
                            text_chunks.append(paragraph.text)

                    # Process images in the DOCX
                    for rel in doc.part.rels.values():
                        if "image" in rel.target_ref:
                            try:
                                image_data = rel.target_part.blob
                                img = Image.open(BytesIO(image_data))
                                text_chunks.append(pytesseract.image_to_string(img))
                            except (UnidentifiedImageError, AttributeError) as e:
                                text_chunks.append("[Corrupted or unsupported image detected, skipping]")
            except PackageNotFoundError:
                raise HTTPException(status_code=400, detail="Error parsing DOCX: Invalid or corrupted file.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error parsing DOCX: {str(e)}")
            return text_chunks

        elif filename.endswith('.csv'):
            # Parse CSV using pandas
            try:
                df = pd.read_csv(StringIO(content.decode('utf-8')))
                text_chunks = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")
            return text_chunks

        elif filename.endswith('.xlsx'):
            # Parse Excel using pandas
            try:
                with BytesIO(content) as excel_stream:
                    df = pd.read_excel(excel_stream)
                    text_chunks = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error parsing Excel: {str(e)}")
            return text_chunks

        elif filename.endswith('.txt'):
            # Parse plain text files
            return content.decode('utf-8').splitlines()

        else:
            raise ValueError(f"Unsupported file type: {filename}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing document {filename}: {str(e)}")
