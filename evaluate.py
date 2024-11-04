import argparse
from glob import glob
from pathlib import Path
from mateval.cdvae.eval_utils import *

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
        crystal_array_list = get_crystals_list(cifs)
        gen_crystals = p_map(lambda x: Crystal(x), crystal_array_list)

        return gen_crystals
    
    def get_test_structures(self):
        csv = pd.read_csv(self.args.testset_path)
        test_crystals = p_map(get_gt_crys_ori, csv['cif'])
        return test_crystals
    
    def evaluate(self):
        gen_crystals = self.get_generated_structures()
        test_crystals = self.get_test_structures()

        n_samples = min(len(gen_crystals), args.n_samples)

        gen_evaluator = GenEval(
            gen_crystals, 
            test_crystals, 
            n_samples=n_samples,
            eval_model_name=self.args.eval_model_name
        )

        print("Calculating evaluation metrics...")
        all_metrics = {}
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

        for k, v in all_metrics.items():
            if k == "comp_valid":
                print(f"Composition Validity: {v:.4f}")
            if k == "struct_valid":
                print(f"Structure Validity: {v:.4f}")
            if k == "wdist_density":
                print(f"EMD Density: {v:.4f}")
            if k == "wdist_num_elems":
                print(f"EMD # Elements: {v:.4f}")
            if k == "cov_recall":
                print(f"Coverage Recall: {v:.4f}")
            if k == "cov_precision":
                print(f"Coverage Precision: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated structures")
    parser.add_argument("--cif_dir", type=str, required=True, help="Directory containing generated cif files")
    parser.add_argument("--testset_path", type=str, help="path to the testset csv", default="data/mp-20/test.csv")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of valid samples to evaluate for earthmover's distance")
    parser.add_argument("--eval_model_name", type=str, help="name of the evaluation model", default="mp20")

    args = parser.parse_args()
    evaluator = Evaluator(args)
    evaluator.evaluate()
