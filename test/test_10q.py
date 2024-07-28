import argparse

from gmft import TableDetector, TableDetectorConfig, TATRTableFormatter, TATRFormatConfig
from gmft.pdf_bindings import PyPDFium2Document
from tabulate import tabulate
from termcolor import colored

detector = TableDetector(
    TableDetectorConfig(
        detector_path="microsoft/table-transformer-detection",
        detector_base_threshold=0.9,
    )
)

# Use the latest TATR model fine-tuned with FinTabNet.c dataset
# https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-fin
formatter = TATRTableFormatter(
    config=TATRFormatConfig(
        formatter_path="microsoft/table-transformer-structure-recognition-v1.1-fin",
        no_timm=False,        
    )
)

def extract_tables_for_page(page):
    tables = detector.extract(page)
    if not tables:
        return

    print(colored(
        f"\nFound {len(tables)} tables on page {page.page_number+1}",
        "green", attrs=["bold"])
    )
    for table in tables:   
        try:
            ft = formatter.extract(table)
            df = ft.df()
        except ValueError as ex:
            print(ex)
            continue

        print(f"Table bbox: {table.rect}, confidence: {table.confidence_score:.3f}")
        # Print DataFrame as a table
        print(tabulate(df, headers='keys', tablefmt='pretty'))


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-p', '--pdf', type=str, required=True)
    args = parser.parse_args()

    doc = PyPDFium2Document(args.pdf)
    try:
        for page in doc:
            extract_tables_for_page(page)
    finally:
        doc.close()
