from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
import os

# 1. Setup paths
SOURCE_PDF = "knowledge_base/basics_of_strength_and_conditioning_manual.pdf"  # Put your PDF name here!
OUTPUT_DIR = "data/knowledge_markdown"


def convert_pdf_to_markdown():
    if not os.path.exists(SOURCE_PDF):
        print(f"File not found: {SOURCE_PDF}")
        return

    print(f"🚀 Docling is analyzing {SOURCE_PDF}...")

    # 2. Initialize the converter
    converter = DocumentConverter()

    # 3. Perform the conversion
    result = converter.convert(SOURCE_PDF)

    # 4. Save as Markdown (The format LLMs love most)
    markdown_output = result.document.export_to_markdown()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "science_base.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_output)

    print(f"✅ Success! Science base saved to {output_path}")


if __name__ == "__main__":
    convert_pdf_to_markdown()