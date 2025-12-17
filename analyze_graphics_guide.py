import pypdf
import os

pdf_path = r"c:\Users\ssdaj\Downloads\graphics-api-performance-guide-2-5.pdf"
output_path = "graphics_guide_full.txt"

def extract_text_from_pdf(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        print(f"Number of pages: {len(reader.pages)}")
        
        # Try to get outline/TOC
        print("\n--- Table of Contents (Outline) ---")
        if reader.outline:
            for item in reader.outline:
                if isinstance(item, list):
                    for subitem in item:
                        if hasattr(subitem, 'title'):
                            print(f"  - {subitem.title}")
                elif hasattr(item, 'title'):
                    print(f"- {item.title}")
        else:
            print("No outline found.")

        print("\n--- Extracting Text ---")
        with open(output_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                f.write(f"\n\n--- Page {i+1} ---\n\n")
                f.write(page_text)
                text += page_text
        
        print(f"Full text saved to {output_path}")
        return text

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

if __name__ == "__main__":
    extract_text_from_pdf(pdf_path)
