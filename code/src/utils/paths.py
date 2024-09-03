from pathlib import Path

ROOT_PATH = (Path(__file__).parent / '..' / '..' / '..').resolve()
GENERATED_DATA_PATH = ROOT_PATH / 'data' / 'generated'
FIGURES_PATH = ROOT_PATH / 'report' / 'figures'
