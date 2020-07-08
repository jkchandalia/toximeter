import sys
import logging
import click
from toxicity import pipelines

logging.basicConfig(
    format='[%(asctime)s|%(module)s.py|%(levelname)s]  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)

@click.command()
@click.option('-d', '--input_data_path',
            required=True,
              type=click.Path(exists=True),
              prompt='Path to the input data file')
@click.option('-o', '--output_model_path',
              type=click.Path(exists=True),
              default=None)
def toxicity_analysis(input_data_path, output_model_path):
    pipelines.build_toxicity_model(input_data_path, output_model_path)
