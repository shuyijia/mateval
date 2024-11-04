# mateval
[CDVAE](https://github.com/txie-93/cdvae) evaluation metrics for structure generation

## Installation
You can use the environment you have created for LLM fine-tuning. Additionally, you need to install the following:

```
pip install matminer p-tqdm smact==2.2.1
```

## Usage
To calculate the metrics, do the following:

```
python evaluate.py --cif_dir [PATH/TO/CIF/FOLDER] --n_samples 200
```

- `cif_dir` is the folder path that contains the generated `.cif` files
- `n_samples` is the number of valid samples to evaluate on. Suppose there are 1000 `.cif` files in the folder, we can take 20% for `n_samples`. This means that you should set `n_samples` to be $0.2 \times 1000 = 200$.

## Metrics
### Validity
`Composition validity`: a structure is compositionally valid if it is charge neutral overall.

`Structure validity`: a structure is structurally valid if all pairwise interatomic distances are $> 0.5$ angstrom. 

### Coverage
`Recall`: measures the percentage of ground truth materials being correctly predicted.

`Precison`: measures the percentage of predicted materials having high quality.

### Earth Mover's Distance (EMD)
In a very general sense, the Earth Mover's Distance measures the amount of effort required to transform one distribution into a target distribution.

`Density`: $g cm^{-3}$

`# elem`: number of unique elements 

For more info on the metrics, please refer to the [original paper](https://arxiv.org/abs/2110.06197)

You can collect the above metrics for your generated structures and put them into a table:

|            | Struct Validity | Comp. Validity | Recall | Precision | Density | # Elem |
|------------|-----------------|----------------|--------|-----------|---------|--------|
| LLaMA-2-7B | 0.964           | 0.933          | 0.969  | 0.960     | 3.85    | 0.960  |

(^ results from [crystal-text-llm](https://github.com/facebookresearch/crystal-text-llm))