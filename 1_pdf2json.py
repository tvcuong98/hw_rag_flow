import os
import re
import json
import fitz  # PyMuPDF

def extract_law_id(filename):
    # Sb_2012_89_2024-04-01_IZ.pdf → 89-2012.json
    m = re.match(r"Sb_(\d{4})_(\d+)_", filename)
    if m:
        return f"{m.group(2)}-{m.group(1)}"
    return os.path.splitext(filename)[0]

def extract_title(lines):
    # Look for first occurrence after "ZÁKON" and "ze dne"
    zak_found = False
    date_found = False
    for idx, line in enumerate(lines):
        if not zak_found and re.search(r"\bZÁKON\b", line, re.IGNORECASE):
            zak_found = True
        elif zak_found and not date_found and re.search(r"\bze dne\b", line, re.IGNORECASE):
            date_found = True
        elif zak_found and date_found and line.strip():
            # Next non-empty line: usually title
            return line.strip()
    # Fallback: first big uppercase title-like line
    for line in lines:
        if len(line) > 8 and line == line.upper() and not re.match(r"^\d+$", line):
            return line.strip()
    return None

def extract_agency(lines):
    # Look for "vyhlašuje:", "ustanovuje:", "schvaluje:" etc. or first 'Parlament ...'
    agency_regex = r"(vyhlašuje|ustanovuje|schvaluje|Parlament[^:]*|Vláda[^:]*|Senát[^:]*):?\s*(.+)?"
    for line in lines:
        m = re.search(agency_regex, line, re.IGNORECASE)
        if m:
            # Return full line or only agency name if possible
            content = m.group(0).strip()
            # Remove excessive trailing punctuation
            return content.rstrip(':.')
    # Fallback: first line with 'Parlament' or 'Vláda'
    for line in lines:
        if "Parlament" in line or "Vláda" in line:
            return line.strip()
    return None

def clean_text(text):
    # Remove pagination, excessive spaces, etc.
    text = re.sub(r"strana\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r" +", " ", text)
    return text

def parse_structure(cleaned):
    # Parse by ČÁST (PART), HLAVA (TITLE), § Paragraphs
    parts = []
    part_pattern = r"(ČÁST\s+[^\n]+)"
    part_matches = [m for m in re.finditer(part_pattern, cleaned)]
    for i, m in enumerate(part_matches):
        start = m.start()
        end = part_matches[i + 1].start() if i + 1 < len(part_matches) else len(cleaned)
        part_block = cleaned[start:end].strip()
        # Parse HLAVA or directly paragraphs if not present
        hlava_pattern = r"(HLAVA\s+[^\n]+)"
        hlava_matches = [hm for hm in re.finditer(hlava_pattern, part_block)]
        hlavas = []
        if hlava_matches:
            for j, hm in enumerate(hlava_matches):
                h_start = hm.start()
                h_end = hlava_matches[j + 1].start() if j + 1 < len(hlava_matches) else len(part_block)
                hlava_block = part_block[h_start:h_end].strip()
                paragraphs = parse_paragraphs(hlava_block)
                hlavas.append({
                    "name": hm.group(1).strip(),
                    "paragraphs": paragraphs
                })
        else:
            # No HLAVA, parse paragraphs for whole part
            paragraphs = parse_paragraphs(part_block)
            hlavas.append({
                "name": None,
                "paragraphs": paragraphs
            })
        parts.append({
            "name": m.group(1).strip(),
            "titles": hlavas
        })
    return parts

def parse_paragraphs(text):
    para_pattern = r"§\s*(\d+[a-zA-Z]*)\.?\s*(.*?)(?=§\s*\d+[a-zA-Z]*\.?|HLAVA|ČÁST|$)"
    paragraphs = []
    for pm in re.finditer(para_pattern, text, re.DOTALL):
        para_name = pm.group(1).strip()
        para_content = pm.group(2).strip()
        subsections = []
        subsec_pattern = r"\((\d+)\)\s*([^\(\)]+)(?=(\(\d+\))|$)"
        sub_matches = list(re.finditer(subsec_pattern, para_content, re.DOTALL))
        if sub_matches:
            for sm in sub_matches:
                sub_name = sm.group(1)
                sub_content = sm.group(2).strip()
                subsections.append({"name": sub_name, "content": sub_content})
        else:
            # Treat the whole paragraph as one subsection if none found
            if para_content:
                subsections = [{"name": "1", "content": para_content}]
        paragraphs.append({
            "name": para_name,
            "subsections": subsections
        })
    return paragraphs

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    cleaned = clean_text(full_text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    # Extract fields
    title = extract_title(lines)
    agency = extract_agency(lines)
    structure = parse_structure(cleaned)
    return title, agency, structure

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".pdf"): continue
        pdf_path = os.path.join(input_folder, fname)
        law_id = extract_law_id(fname)
        print(f"Processing {fname} (law_id: {law_id})...")
        title, agency, structure = process_pdf(pdf_path)
        # Compose JSON
        json_struct = {
            "law_id": law_id,
            "title": title,
            "agency": agency,
            "structure": structure
        }
        out_path = os.path.join(output_folder, f"{law_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_struct, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, help="Folder containing input PDFs")
    parser.add_argument("--output_folder", required=True, help="Folder to save output JSONs")
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder)
