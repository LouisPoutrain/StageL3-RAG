from grobid_client.grobid_client import GrobidClient

# Initialisation du client Grobid
client = GrobidClient(config_path="./config.json")

# Traitement des documents
client.process(
    service='processFulltextDocument',
    input_path='pdfs',
    output='results',
    consolidate_citations=True,
    tei_coordinates=True,
    force=True
)