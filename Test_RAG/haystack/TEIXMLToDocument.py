from bs4 import BeautifulSoup
from haystack.core.component import component
from haystack.dataclasses import Document
from typing import List, Dict, Any
@component
class TEIXMLToDocument:
    @component.output_types(documents=List[Document])
    def run(self, sources: List[str]) -> Dict[str, Any]:
        documents = []
        print(f"Traitement de {len(sources)} fichiers TEI XML")
        
        for source in sources:
            try:
                with open(source, 'r', encoding='utf-8') as file:
                    content = file.read()
                    soup = BeautifulSoup(content, 'xml')
                    
                    # Extraction du titre
                    title = soup.find('titleStmt')
                    title_text = title.get_text(separator=' ', strip=True) if title else 'Sans titre'
                    print(f"Titre extrait : {title_text}")
                    
                    # Extraction de l'abstract
                    abstract = soup.find('abstract')
                    abstract_text = abstract.get_text(separator=' ', strip=True) if abstract else ''
                    
                    # Extraction du corps
                    body = soup.find('body')
                    if body:
                        sections = body.find_all('div', type=['section', 'subsection'])
                        if not sections:
                            # Si aucune section n'est trouvée, utiliser le corps entier
                            sections = [body]
                        
                        for i, section in enumerate(sections):
                            section_text = section.get_text(separator=' ', strip=True)
                            if section_text: 
                                documents.append(Document(
                                    content=section_text,
                                    meta={
                                        'source': str(source),
                                        'title': title_text,
                                        'abstract': abstract_text,
                                        'section': i
                                    }
                                ))
                    else:
                        print(f"Aucun corps de texte trouvé dans {source}")
            except Exception as e:
                print(f"Erreur lors du traitement de {source}: {e}")
        
        print(f"Nombre de documents extraits : {len(documents)}")
        return {"documents": documents}