from pdfminer.high_level import extract_text
import tempfile

def extract_text_from_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:

        tmp_file.write(uploaded_file.read())

        path = tmp_file.name

    text = extract_text(path)

    return text
