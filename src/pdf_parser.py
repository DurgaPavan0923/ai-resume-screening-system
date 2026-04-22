import fitz  # PyMuPDF

def parse_pdf(uploaded_file):
    try:
        text = ""
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

        for page in pdf:
            text += page.get_text()

        return text.strip()

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
