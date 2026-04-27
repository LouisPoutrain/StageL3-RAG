import pandas as pd
import requests
import os
import time
from bs4 import BeautifulSoup
import urllib.parse
import re
import json
from urllib.error import HTTPError
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import xml.etree.ElementTree as ET
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

class ScientificPaperFinder:
    def __init__(self, xlsx_path, output_folder="downloaded_papers2", use_browser=True):
        """
        Initialise le service de recherche d'articles scientifiques.
        
        Args:
            xlsx_path (str): Chemin vers le fichier Excel contenant les titres
            output_folder (str): Dossier où seront sauvegardés les PDF
            use_browser (bool): Utiliser Selenium pour accéder aux sources qui nécessitent un navigateur
        """
        self.xlsx_path = xlsx_path
        self.output_folder = output_folder
        self.use_browser = use_browser
        self.browser = None
        
        # Créer le dossier de sortie s'il n'existe pas
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # Sources d'articles en accès ouvert
        self.sources = {
            "unpaywall": "https://api.unpaywall.org/v2/search?query={}&email=your_email@example.com",
            "core": "https://api.core.ac.uk/v3/search/works?q={}&limit=1",
            "arxiv": "http://export.arxiv.org/api/query?search_query=ti:\"{}\"&start=0&max_results=1",
            "pmc": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={}",
            "pubmed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}",
            "biorxiv": "https://api.biorxiv.org/details/biorxiv/{}",
            "europe_pmc": "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={}&format=json&resultType=core",
            "doaj": "https://doaj.org/api/search/articles/{}",
            "plos": "https://api.plos.org/search?q=title:\"{}\"&fl=id,title,journal,eissn,author,abstract,publication_date&wt=json",
            "crossref": "https://api.crossref.org/works?query.title={}&rows=1"
        }
        
        # Initialiser le navigateur si nécessaire
        if self.use_browser:
            self._setup_browser()
            
    def _setup_browser(self):
        """Configure le navigateur headless pour les sources qui nécessitent JavaScript"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Créer une instance du navigateur
            self.browser = webdriver.Chrome(options=chrome_options)
            print("Navigateur headless initialisé avec succès")
        except Exception as e:
            print(f"⚠️ Impossible d'initialiser le navigateur: {e}")
            print("Certaines sources ne seront pas accessibles")
            self.browser = None
        
    def clean_filename(self, title):
        """Nettoie le titre pour en faire un nom de fichier valide"""
        return re.sub(r'[\\/*?:"<>|]', "", title)[:100]
        
    def load_data(self):
        """Charge les données depuis le fichier Excel"""
        try:
            self.df = pd.read_excel(self.xlsx_path)
            if 'Title' not in self.df.columns:
                raise ValueError("Le fichier Excel doit contenir une colonne 'Title'")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement du fichier Excel: {e}")
            return False
            
    def search_unpaywall(self, title):
        """Recherche sur Unpaywall"""
        try:
            query_url = self.sources["unpaywall"].format(urllib.parse.quote(title))
            response = requests.get(query_url)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("results", []):
                    if "best_oa_location" in item and item["best_oa_location"]:
                        pdf_url = item["best_oa_location"].get("url_for_pdf")
                        if pdf_url:
                            return pdf_url
            return None
        except Exception as e:
            print(f"Erreur avec Unpaywall: {e}")
            return None
            
    def search_core(self, title):
        """Recherche sur CORE"""
        try:
            headers = {"Authorization": "Bearer cjY4ywGPJ9ZVso0fb8m2RBFNEIWSadKv"} 
            query_url = self.sources["core"].format(urllib.parse.quote(title))
            response = requests.get(query_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("results", []):
                    if "downloadUrl" in item:
                        return item["downloadUrl"]
            return None
        except Exception as e:
            print(f"Erreur avec CORE: {e}")
            return None
            
    def search_arxiv(self, title):
        """Recherche sur arXiv"""
        try:
            query_url = self.sources["arxiv"].format(urllib.parse.quote(title))
            response = requests.get(query_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "xml")
                entries = soup.find_all("entry")
                if entries:
                    pdf_link = entries[0].find("link", {"title": "pdf"})
                    if pdf_link:
                        return pdf_link.get("href")
            return None
        except Exception as e:
            print(f"Erreur avec arXiv: {e}")
            return None
    
    def search_pubmed_central(self, title):
        """Recherche sur PubMed Central (PMC) pour des articles en accès libre"""
        try:
            # Recherche d'abord l'ID de l'article
            search_url = self.sources["pmc"].format(urllib.parse.quote(title))
            response = requests.get(search_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "xml")
                id_elements = soup.find_all("Id")
                if id_elements and len(id_elements) > 0:
                    pmc_id = id_elements[0].text
                    # Utiliser l'ID pour obtenir le lien du PDF
                    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
                    # Vérifier si le PDF existe
                    check_response = requests.head(pdf_url)
                    if check_response.status_code == 200:
                        return pdf_url
            return None
        except Exception as e:
            print(f"Erreur avec PubMed Central: {e}")
            return None
    
    def __del__(self):
        """Ferme le navigateur lorsque l'objet est détruit"""
        if self.browser:
            try:
                self.browser.quit()
                print("Navigateur fermé")
            except:
                pass
    
    def extract_doi_from_text(self, text):
        """Extrait un DOI du texte"""
        doi_pattern = r"10.\d{4,9}/[-._;()/:A-Z0-9]+"
        match = re.search(doi_pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
        return None
    
    def search_doaj(self, title):
        """Recherche dans le Directory of Open Access Journals"""
        try:
            query_url = self.sources["doaj"].format(urllib.parse.quote(title))
            response = requests.get(query_url)
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) > 0:
                    for result in data["results"]:
                        if "bibjson" in result:
                            bibjson = result["bibjson"]
                            if "link" in bibjson:
                                for link in bibjson["link"]:
                                    if "type" in link and link["type"] == "fulltext":
                                        return link.get("url")
            return None
        except Exception as e:
            print(f"Erreur avec DOAJ: {e}")
            return None
    
    def search_plos(self, title):
        """Recherche dans la Public Library of Science (PLOS)"""
        try:
            query_url = self.sources["plos"].format(urllib.parse.quote(title))
            response = requests.get(query_url)
            if response.status_code == 200:
                data = response.json()
                if "response" in data and "docs" in data["response"] and len(data["response"]["docs"]) > 0:
                    doc = data["response"]["docs"][0]
                    if "id" in doc:
                        doi = doc["id"]
                        pdf_url = f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable"
                        check_response = requests.head(pdf_url)
                        if check_response.status_code == 200:
                            return pdf_url
            return None
        except Exception as e:
            print(f"Erreur avec PLOS: {e}")
            return None
    
    def search_crossref(self, title):
        """Recherche dans CrossRef pour obtenir le DOI"""
        try:
            query_url = self.sources["crossref"].format(urllib.parse.quote(title))
            response = requests.get(query_url)
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "items" in data["message"] and len(data["message"]["items"]) > 0:
                    item = data["message"]["items"][0]
                    if "DOI" in item:
                        return item["DOI"]
            return None
        except Exception as e:
            print(f"Erreur avec CrossRef: {e}")
            return None
    
    def search_by_doi(self, doi_str):
        """Tente de récupérer un PDF à partir d'un DOI"""
        try:
            # Essayer Unpaywall avec le DOI
            url = f"https://api.unpaywall.org/v2/{doi_str}?email=raphael.malidin@gmail.com"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get("is_oa") and data.get("best_oa_location") and data["best_oa_location"].get("url_for_pdf"):
                    return data["best_oa_location"]["url_for_pdf"]
                
            # Essayer le format standard pour ResearchGate
            pdf_url = f"https://www.researchgate.net/profile/publication/{doi_str}/fulltext.pdf"
            response = requests.head(pdf_url)
            if response.status_code == 200:
                return pdf_url
                
            return None
        except Exception as e:
            print(f"Erreur avec la recherche par DOI: {e}")
            return None
    
    def search_google_scholar_selenium(self, title):
        """Utilise Selenium pour rechercher sur Google Scholar"""
        if not self.browser:
            print("Le navigateur n'est pas disponible pour la recherche Google Scholar")
            return None
            
        try:
            print("Tentative de recherche automatique sur Google Scholar...")
            url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(title)}"
            self.browser.get(url)
            
            # Attendre que les résultats se chargent
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "gs_r"))
            )
            
            # Chercher des liens PDF
            pdf_links = self.browser.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
            if pdf_links:
                pdf_url = pdf_links[0].get_attribute("href")
                return pdf_url
                
            # Chercher des liens DOI
            all_links = self.browser.find_elements(By.TAG_NAME, "a")
            for link in all_links:
                href = link.get_attribute("href")
                if href and "doi.org" in href:
                    doi_str = self.extract_doi_from_text(href)
                    if doi_str:
                        return self.search_by_doi(doi_str)
            
            return None
        except Exception as e:
            print(f"Erreur avec Google Scholar Selenium: {e}")
            return None
    
    def search_pubmed_open_access(self, title):
        """Recherche spécifiquement des articles en accès libre sur PubMed"""
        try:
            search_term = f"{title} AND open access[filter]"
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(search_term)}"
            response = requests.get(search_url)
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                id_elements = root.findall(".//Id")
                
                if id_elements and len(id_elements) > 0:
                    pubmed_id = id_elements[0].text
                    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pubmed_id}&retmode=xml"
                    fetch_response = requests.get(fetch_url)
                    
                    if fetch_response.status_code == 200:
                        fetch_root = ET.fromstring(fetch_response.content)
                        article_ids = fetch_root.findall(".//ArticleId")
                        
                        for article_id in article_ids:
                            if article_id.get("IdType") == "doi":
                                doi = article_id.text
                                return self.search_by_doi(doi)
            return None
        except Exception as e:
            print(f"Erreur avec PubMed Open Access: {e}")
            return None
    
    def search_biorxiv(self, title):
        """Recherche sur bioRxiv pour des prépublications en biologie"""
        try:
            query_url = self.sources["biorxiv"].format(urllib.parse.quote(title))
            response = requests.get(query_url)
            if response.status_code == 200:
                data = response.json()
                if "collection" in data and len(data["collection"]) > 0:
                    paper = data["collection"][0]
                    if "doi" in paper:
                        doi = paper["doi"]
                        pdf_url = f"https://www.biorxiv.org/content/{doi}v1.full.pdf"
                        check_response = requests.head(pdf_url)
                        if check_response.status_code == 200:
                            return pdf_url
            return None
        except Exception as e:
            print(f"Erreur avec bioRxiv: {e}")
            return None
    
    def search_europe_pmc(self, title):
        """Recherche sur Europe PMC pour des articles en accès libre"""
        try:
            query_url = self.sources["europe_pmc"].format(urllib.parse.quote(title))
            response = requests.get(query_url)
            if response.status_code == 200:
                data = response.json()
                if "resultList" in data and "result" in data["resultList"] and len(data["resultList"]["result"]) > 0:
                    result = data["resultList"]["result"][0]
                    if "pmcid" in result and result["isOpenAccess"] == "Y":
                        pmcid = result["pmcid"]
                        pdf_url = f"https://europepmc.org/articles/{pmcid}/pdf"
                        check_response = requests.head(pdf_url)
                        if check_response.status_code == 200:
                            return pdf_url
            return None
        except Exception as e:
            print(f"Erreur avec Europe PMC: {e}")
            return None
            
    def search_google_scholar(self, title):
        """
        Recherche via Google Scholar
        
        """
        print(f"Pour l'article '{title}', vous pourriez essayer de rechercher manuellement sur:")
        print(f"Google Scholar: https://scholar.google.com/scholar?q={urllib.parse.quote(title)}")
        print(f"PubMed: https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(title)}")
        print(f"bioRxiv: https://www.biorxiv.org/search/{urllib.parse.quote(title)}")
        print(f"Europe PMC: https://europepmc.org/search?query={urllib.parse.quote(title)}")
        print(f"SciHub: https://sci-hub.se/")
        return None
            
    def download_pdf(self, url, filename):
        """Télécharge le PDF depuis l'URL"""
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                filepath = os.path.join(self.output_folder, f"{filename}.pdf")
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return filepath
            return None
        except Exception as e:
            print(f"Erreur lors du téléchargement: {e}")
            return None
            
    def process_all(self):
        if not self.load_data():
            return

        grobid_url = "http://localhost:8070/api/processFulltextDocument"
        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)

        try:
            response = session.get("http://localhost:8070/api/isalive", timeout=5)
            response.raise_for_status()
            print("✅ GROBID server is up and running.")
        except RequestException as e:
            print(f"❌ GROBID server is not running: {e}")
            return

        os.makedirs("results", exist_ok=True)
        results = []

        for i, row in self.df.iterrows():
            title = row['Title']
            filename_base = self.clean_filename(title)
            pdf_path = os.path.join(self.output_folder, f"{filename_base}.pdf")
            xml_path = os.path.join("results", f"{filename_base}.grobid.tei.xml")

            print(f"\n{'='*50}\nTitre : {title}\n{'='*50}")

            # --- Vérifier si PDF existe déjà ---
            if os.path.exists(pdf_path):
                print(f"📄 PDF déjà existant, on saute le téléchargement : {pdf_path}")
            else:
                # Recherche du PDF
                pdf_url = None
                source = None
                for search_func, src_name in [
                    *[(self.search_pubmed_central, "PubMed Central"),
                    (self.search_pubmed_open_access, "PubMed Open Access"),
                    (self.search_biorxiv, "BioRxiv"),
                    (self.search_plos, "PLOS"),
                    (self.search_europe_pmc, "Europe PMC"),
                    (self.search_doaj, "DOAJ")],
                    *[(self.search_unpaywall, "Unpaywall"),
                    (self.search_core, "CORE"),
                    (self.search_arxiv, "arXiv")]
                ]:
                    print(f"🔍 Recherche sur {src_name}...")
                    pdf_url = search_func(title)
                    if pdf_url:
                        source = src_name
                        print(f"✅ Trouvé sur {src_name}")
                        break
                    print(f"❌ Non trouvé sur {src_name}")

                if not pdf_url:
                    doi_str = self.search_crossref(title)
                    if doi_str:
                        print(f"🔎 DOI trouvé : {doi_str}")
                        pdf_url = self.search_by_doi(doi_str)

                if not pdf_url and self.use_browser and self.browser:
                    pdf_url = self.search_google_scholar_selenium(title)

                if pdf_url:
                    path = self.download_pdf(pdf_url, filename_base)
                    if path:
                        print(f"✅ PDF téléchargé : {path}")
                    else:
                        print(f"⚠️ Échec du téléchargement")
                        results.append({"Titre": title, "Fichier PDF": "Erreur", "Fichier XML": "Non généré", "Statut": "Échec téléchargement"})
                        continue
                else:
                    print("❌ Article non trouvé")
                    results.append({"Titre": title, "Fichier PDF": "Non trouvé", "Fichier XML": "Non généré", "Statut": "Introuvable"})
                    continue

            # --- Vérifier si XML déjà généré ---
            if os.path.exists(xml_path):
                print(f"📝 XML déjà existant, on saute GROBID : {xml_path}")
                results.append({
                    "Titre": title,
                    "Fichier PDF": os.path.basename(pdf_path),
                    "Fichier XML": os.path.basename(xml_path),
                    "Statut": "Déjà traité"
                })
                continue

            # --- Traitement GROBID ---
            try:
                with open(pdf_path, 'rb') as f:
                    response = session.post(grobid_url, files={'input': f}, timeout=(5, 90))
                    response.raise_for_status()
                with open(xml_path, 'w', encoding='utf-8') as out:
                    out.write(response.text)
                print(f"✅ GROBID terminé : {xml_path}")
                results.append({
                    "Titre": title,
                    "Fichier PDF": os.path.basename(pdf_path),
                    "Fichier XML": os.path.basename(xml_path),
                    "Statut": "Succès"
                })
            except Exception as e:
                print(f"❌ Erreur GROBID : {e}")
                results.append({
                    "Titre": title,
                    "Fichier PDF": os.path.basename(pdf_path),
                    "Fichier XML": "Erreur",
                    "Statut": "GROBID échoué"
                })

            time.sleep(2)

        # === Export des résultats ===
        with open("Résultat_extraction.txt", "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"{r['Titre']} | {r['Fichier PDF']} | {r['Fichier XML']}\n")

        print(f"\n📄 Résultat_extraction.txt généré avec {len(results)} entrées.")
        return results





if __name__ == "__main__":
    finder = ScientificPaperFinder(
        "FinalRawData.xlsx",
        use_browser=True  
    )

    with open("Résultat_extraction.txt", "w", encoding="utf-8") as f:
        f.write(finder.process_all())
    # Traiter tous les articles du fichier Excel
    finder.process_all()
    
    if finder.browser:
        finder.browser.quit()