
#%% copy data
"""
This script performs preprocessing, quality control, normalization, and clustering analysis on two single-cell RNA-seq datasets (PBMC 3k v1 and PBMC 10k v3) using Scanpy. The workflow includes:

1. Downloading and extracting 10x Genomics PBMC datasets if not already present.
2. Loading the datasets into AnnData objects.
3. Filtering cells and genes based on quality control metrics (e.g., minimum genes per cell, minimum cells per gene).
4. Calculating mitochondrial gene content and other QC metrics.
5. Visualizing QC metrics using violin plots.
6. Identifying and removing outlier cells based on mitochondrial content, total counts, and number of genes.
7. Selecting highly variable genes using Pearson residuals.
8. Visualizing gene variability and highlighting known marker genes.
9. Subsetting datasets to highly variable genes.
10. Storing raw and normalized counts in AnnData layers.
11. Normalizing data using Pearson residuals.
12. Performing dimensionality reduction (PCA and t-SNE).
13. Computing neighborhood graphs and performing Leiden clustering.
14. Visualizing t-SNE embeddings colored by cluster and marker gene expression.
15. Concatenating the two datasets, annotating batch information, and saving the combined AnnData object to an HDF5 file.

References:
- Scanpy documentation: https://scanpy.readthedocs.io/
- PBMC 3k tutorial: https://scanpy.readthedocs.io/en/stable/tutorials/experimental/pearson_residuals.html
"""
import os

if not os.path.exists("tutorial_data"):
    print("Folder 'tutorial_data' does not exist.")
    !mkdir tutorial_data
    !mkdir tutorial_data/pbmc3k_v1
    !mkdir tutorial_data/pbmc10k_v3

    !wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -O tutorial_data/pbmc3k_v1.tar.gz
    !cd tutorial_data; tar -xzf pbmc3k_v1.tar.gz -C pbmc3k_v1 --strip-components 2

    !wget https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.tar.gz -O tutorial_data/pbmc10k_v3.tar.gz
    !cd tutorial_data; tar -xzf pbmc10k_v3.tar.gz -C pbmc10k_v3 --strip-components 1

else:
    print("Folder 'tutorial_data' already exists.")


# from https://scanpy.readthedocs.io/en/stable/tutorials/experimental/pearson_residuals.html
#%%
import numpy as np
import scanpy as sc
import os
import matplotlib.pyplot as plt

#%%

adata_pbmc3k = sc.read_10x_mtx("tutorial_data/pbmc3k_v1/", cache=True)
adata_pbmc10k = sc.read_10x_mtx("tutorial_data/pbmc10k_v3/", cache=True)

adata_pbmc3k.uns["name"] = "PBMC 3k (v1)"
adata_pbmc10k.uns["name"] = "PBMC 10k (v3)"


#%%
# marker genes from table in pbmc3k tutorial
markers = [
    "IL7R",
    "LYZ",
    "CD14",
    "MS4A1",
    "CD8A",
    "GNLY",
    "NKG7",
    "FCGR3A",
    "MS4A7",
    "FCER1A",
    "CST3",
    "PPBP",
]

# %% basic filtering

for adata in [adata_pbmc3k, adata_pbmc10k]:
    adata.var_names_make_unique()
    print(adata.uns["name"], ": data shape:", adata.shape)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
# %%
for adata in [adata_pbmc3k, adata_pbmc10k]:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
# %%
for adata in [adata_pbmc3k, adata_pbmc10k]:
    print(adata.uns["name"], ":")
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
    )
# %%
# define outliers and do the filtering for the 3k dataset
adata_pbmc3k.obs["outlier_mt"] = adata_pbmc3k.obs.pct_counts_mt > 5
adata_pbmc3k.obs["outlier_total"] = adata_pbmc3k.obs.total_counts > 5000
adata_pbmc3k.obs["outlier_ngenes"] = adata_pbmc3k.obs.n_genes_by_counts > 2500

print(
    "%u cells with high %% of mitochondrial genes"
    % (sum(adata_pbmc3k.obs["outlier_mt"]))
)
print("%u cells with large total counts" % (sum(adata_pbmc3k.obs["outlier_total"])))
print("%u cells with large number of genes" % (sum(adata_pbmc3k.obs["outlier_ngenes"])))

adata_pbmc3k = adata_pbmc3k[~adata_pbmc3k.obs["outlier_mt"], :]
adata_pbmc3k = adata_pbmc3k[~adata_pbmc3k.obs["outlier_total"], :]
adata_pbmc3k = adata_pbmc3k[~adata_pbmc3k.obs["outlier_ngenes"], :]
sc.pp.filter_genes(adata_pbmc3k, min_cells=1)
# %%
# define outliers and do the filtering for the 10k dataset
adata_pbmc10k.obs["outlier_mt"] = adata_pbmc10k.obs.pct_counts_mt > 20
adata_pbmc10k.obs["outlier_total"] = adata_pbmc10k.obs.total_counts > 25000
adata_pbmc10k.obs["outlier_ngenes"] = adata_pbmc10k.obs.n_genes_by_counts > 6000

print(
    "%u cells with high %% of mitochondrial genes"
    % (sum(adata_pbmc10k.obs["outlier_mt"]))
)
print("%u cells with large total counts" % (sum(adata_pbmc10k.obs["outlier_total"])))
print(
    "%u cells with large number of genes" % (sum(adata_pbmc10k.obs["outlier_ngenes"]))
)

adata_pbmc10k = adata_pbmc10k[~adata_pbmc10k.obs["outlier_mt"], :]
adata_pbmc10k = adata_pbmc10k[~adata_pbmc10k.obs["outlier_total"], :]
adata_pbmc10k = adata_pbmc10k[~adata_pbmc10k.obs["outlier_ngenes"], :]
sc.pp.filter_genes(adata_pbmc10k, min_cells=1)
# %% Compute 2000 variable genes with Pearson residuals

for adata in [adata_pbmc3k, adata_pbmc10k]:
    sc.experimental.pp.highly_variable_genes(
        adata, flavor="pearson_residuals", n_top_genes=2000
    )
# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, adata in zip(axes, [adata_pbmc3k, adata_pbmc10k]):
    hvgs = adata.var["highly_variable"]

    ax.scatter(
        adata.var["mean_counts"], adata.var["residual_variances"], s=3, edgecolor="none"
    )
    ax.scatter(
        adata.var["mean_counts"][hvgs],
        adata.var["residual_variances"][hvgs],
        c="tab:red",
        label="selected genes",
        s=3,
        edgecolor="none",
    )
    ax.scatter(
        adata.var["mean_counts"][np.isin(adata.var_names, markers)],
        adata.var["residual_variances"][np.isin(adata.var_names, markers)],
        c="k",
        label="known marker genes",
        s=10,
        edgecolor="none",
    )
    ax.set_xscale("log")
    ax.set_xlabel("mean expression")
    ax.set_yscale("log")
    ax.set_ylabel("residual variance")
    ax.set_title(adata.uns["name"])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
plt.legend()
# %%
adata_pbmc3k = adata_pbmc3k[:, adata_pbmc3k.var["highly_variable"]]
adata_pbmc10k = adata_pbmc10k[:, adata_pbmc10k.var["highly_variable"]]
# %%
# keep raw and depth-normalized counts for later
adata_pbmc3k.layers["raw"] = adata_pbmc3k.X.copy()
adata_pbmc3k.layers["sqrt_norm"] = np.sqrt(
    sc.pp.normalize_total(adata_pbmc3k, inplace=False)["X"]
)

adata_pbmc10k.layers["raw"] = adata_pbmc10k.X.copy()
adata_pbmc10k.layers["sqrt_norm"] = np.sqrt(
    sc.pp.normalize_total(adata_pbmc10k, inplace=False)["X"]
)
# %%
# pearson residuals
for adata in [adata_pbmc3k, adata_pbmc10k]:
    sc.experimental.pp.normalize_pearson_residuals(adata)
# %% Compute PCA and t-SNE
for adata in [adata_pbmc3k, adata_pbmc10k]:
    sc.pp.pca(adata, n_comps=50)
    n_cells = len(adata)
    sc.tl.tsne(adata, use_rep="X_pca")


#%% # Leiden clustering
for adata in [adata_pbmc3k, adata_pbmc10k]:
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
# %%
for adata in [adata_pbmc3k, adata_pbmc10k]:
    print(adata.uns["name"], ":")
    sc.pl.tsne(adata, color=["leiden"], cmap="tab20")
    sc.pl.tsne(adata, color=markers, layer="sqrt_norm")
# %%

from scanpy import anndata 
xx = anndata.concat((adata_pbmc3k,adata_pbmc10k))
xx.obs['highlow'] = xx.obs['n_genes']>500
xx.obs['batch']='3k'
xx.obs.iloc[adata_pbmc3k.shape[0]:, xx.obs.columns.get_loc('batch')] = '10k'

sc.write('demo.h5ad', xx)

# %%
