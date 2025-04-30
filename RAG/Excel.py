from openpyxl import Workbook

def parse_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = content.strip().split("="*80)
    rows = []

    for chunk in chunks:
        lines = chunk.strip().splitlines()
        data = {"Fichier": "", "Titre": "", "Date": "", "Auteurs": "", "Résumé": ""}
        current_field = None

        for line in lines:
            if line.startswith("FILE:"):
                data["Fichier"] = line.replace("FILE:", "").strip()
            elif line.startswith("TITLE:"):
                data["Titre"] = line.replace("TITLE:", "").strip()
            elif line.startswith("DATE:"):
                data["Date"] = line.replace("DATE:", "").strip()
            elif line.startswith("AUTHORS:"):
                data["Auteurs"] = line.replace("AUTHORS:", "").strip()
            elif line.startswith("ABSTRACT:"):
                current_field = "Résumé"
                data["Résumé"] = ""
            elif current_field == "Résumé":
                data["Résumé"] += line.strip() + " "

        rows.append(data)
    return rows

def write_to_excel(rows, output_file):
    wb = Workbook()
    ws = wb.active
    ws.title = "Résumé articles"

    headers = ["Fichier", "Titre", "Date", "Auteurs", "Résumé"]
    ws.append(headers)

    for row in rows:
        ws.append([row[h] for h in headers])

    wb.save(output_file)
    print(f"✅ Excel enregistré sous : {output_file}")

# === Utilisation ===
if __name__ == "__main__":
    input_txt = "output_chunks.txt"
    output_excel = "articles_grobid.xlsx"

    data = parse_chunks(input_txt)
    write_to_excel(data, output_excel)
