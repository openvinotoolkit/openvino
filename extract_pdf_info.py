import os
from pypdf import PdfReader

pdf_dir = r"C:\Users\ssdaj\openvino\docs"
keywords = [
    "Gen9", "UHD 620", "Architecture",
    "Compute Shader", "CS",
    "L3", "SLM", "Shared Local Memory", "Bandwidth",
    "FP16", "Half-Precision",
    "Thread", "EU", "Execution Unit", "Occupancy", "Saturation"
]

def extract_info(pdf_path):
    print(f"\n{'='*50}\nProcessing: {os.path.basename(pdf_path)}\n{'='*50}")
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"Total Pages: {num_pages}")

        # Read first 3 pages for TOC/Intro
        print("\n--- Introduction / TOC (First 3 Pages) ---")
        for i in range(min(3, num_pages)):
            text = reader.pages[i].extract_text()
            print(f"\n[Page {i+1}]\n{text[:1000]}...") # Print first 1000 chars

        # Search for keywords in the rest
        print("\n--- Relevant Sections ---")
        for i in range(num_pages):
            text = reader.pages[i].extract_text()
            found_keywords = [k for k in keywords if k.lower() in text.lower()]
            if found_keywords:
                print(f"\n[Page {i+1}] Found keywords: {found_keywords}")
                # Print context around keywords? Or just the whole page text if it's dense?
                # Let's print the first 500 chars and then snippets around keywords
                # For now, just print the text if it has multiple keywords
                if len(found_keywords) > 2:
                    print(text)
                else:
                    print(text[:500] + "\n...\n")

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
for f in pdf_files:
    extract_info(os.path.join(pdf_dir, f))
