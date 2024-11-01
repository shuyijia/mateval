from glob import glob
from pathlib import Path
from cdvae.eval_utils import *

class Evaluator:
    """
    evaluate generated structures with respect to the target structures
    """
    def __init__(self, args):
        self.args = args

    def get_generated_structures(self):
        cif_dir = Path(self.args.cif_dir)
        cifs = glob(f"{cif_dir}/*.cif")

        print(f"Found {len(cifs)} cif files in {cif_dir}")

        # get crystals for generated structures
        