# cxg
## CellXGene heavily relies on scanpy, a single cell analysis python package

### Installation

To install `cellxgene` and `scanpy` using [uv](https://github.com/astral-sh/uv):

```bash
uv venv .venv
source .venv/bin/activate
uv pip install cellxgene scanpy
```

For more details, see the [cellxgene documentation](https://cellxgene.cziscience.com/) and [scanpy documentation](https://scanpy.readthedocs.io/).



### References

For reference material look at the links below

* [pearson_residuals](https://scanpy.readthedocs.io/en/stable/tutorials/experimental/pearson_residuals.html)
* [scverse/scanpy: Single-cell analysis in Python. Scales to >100M cells.](https://github.com/scverse/scanpy/tree/main)


### Example code, format_ex.py

Most of the code originates from [pearson_residuals tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/experimental/pearson_residuals.html). Some examples have been included to add additional filters

```python
xx = anndata.concat((adata_pbmc3k,adata_pbmc10k))
xx.obs['highlow'] = xx.obs['n_genes']>500
xx.obs['batch']='3k'
xx.obs.iloc[adata_pbmc3k.shape[0]:, xx.obs.columns.get_loc('batch')] = '10k'
sc.write('demo.h5ad', xx)
```