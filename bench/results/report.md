
### MCPToolBench++ routing (n=1509)

| router | top1 | recall@5 | server@1 | category@1 | top1@50%cov | shortlist_hit | shortlist_size | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.535 | 0.712 | 0.598 | 0.695 | 0.832 | - | - | 0 | 0 |
| dense-minilm | 0.547 | 0.777 | 0.682 | 0.895 | 0.707 | - | - | 5 | 10 |
| dense-mpnet | 0.557 | 0.794 | 0.652 | 0.889 | 0.703 | - | - | 10 | 16 |
| hybrid-rrf | 0.624 | 0.796 | 0.702 | 0.848 | 0.833 | - | - | 5 | 9 |
| tooljev-encoder-hier | 0.181 | 0.341 | 0.307 | 0.561 | 0.292 | 0.335 | 4.74 | 743 | 1457 |
| tooljev-encoder-rerank | 0.193 | 0.587 | 0.447 | 0.627 | 0.227 | 0.567 | 4.77 | 162 | 246 |
| tooljev-encoder | 0.624 | 0.796 | 0.702 | 0.848 | 0.656 | 0.570 | 2.81 | 74 | 97 |
| hybrid-bge | 0.698 | 0.845 | 0.732 | 0.927 | 0.973 | - | - | 8 | 10 |
| tooljev-hosted-fit | 0.700 | 0.850 | 0.700 | 0.975 | 0.750 | 0.650 | 2.33 | 275 | 383 |
| tooljev-hosted-rerank | 0.825 | 0.955 | 0.850 | 0.990 | 0.980 | 0.925 | 1.48 | 284 | 345 |

### LiveMCPBench routing (n=94)

| router | top1 | recall@5 | server@1 | top1@50%cov | shortlist_hit | shortlist_size | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.266 | 0.306 | 0.479 | 0.383 | - | - | 1 | 2 |
| dense-minilm | 0.287 | 0.350 | 0.532 | 0.468 | - | - | 4 | 9 |
| dense-mpnet | 0.309 | 0.362 | 0.585 | 0.383 | - | - | 11 | 20 |
| hybrid-rrf | 0.309 | 0.405 | 0.564 | 0.468 | - | - | 5 | 15 |
| tooljev-encoder-hier | 0.106 | 0.076 | 0.149 | 0.106 | 0.128 | 3.43 | 4894 | 6877 |
| tooljev-encoder-rerank | 0.245 | 0.284 | 0.447 | 0.362 | 0.468 | 4.77 | 168 | 390 |
| tooljev-encoder | 0.309 | 0.405 | 0.564 | 0.468 | 0.457 | 3.21 | 93 | 181 |
| hybrid-bge | 0.404 | 0.424 | 0.660 | 0.511 | - | - | 9 | 15 |
| tooljev-hosted-fit | 0.375 | 0.404 | 0.625 | 0.600 | 0.600 | 2.35 | 288 | 402 |
| tooljev-hosted-rerank | 0.543 | 0.499 | 0.723 | 0.638 | 0.723 | 2.21 | 292 | 468 |

### When2Call routing (n=250)

| router | top1 | recall@5 | server@1 | top1@50%cov | shortlist_hit | shortlist_size | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.716 | 1.000 | 0.716 | 0.752 | - | - | 0 | 0 |
| dense-minilm | 0.880 | 1.000 | 0.880 | 0.920 | - | - | 3 | 5 |
| dense-mpnet | 0.908 | 1.000 | 0.908 | 0.928 | - | - | 11 | 22 |
| hybrid-rrf | 0.792 | 1.000 | 0.792 | 1.000 | - | - | 4 | 9 |
| tooljev-encoder-hier | 0.684 | 0.976 | 0.684 | 0.888 | 0.948 | 1.80 | 55 | 172 |
| tooljev-encoder-rerank | 0.667 | 1.000 | 0.667 | 0.893 | 0.980 | 2.19 | 40 | 114 |
| tooljev-encoder | 0.792 | 1.000 | 0.792 | 0.944 | 0.948 | 1.80 | 28 | 77 |
| hybrid-bge | 0.916 | 1.000 | 0.916 | 1.000 | - | - | 8 | 9 |
| tooljev-hosted-fit | 0.900 | 1.000 | 0.900 | 1.000 | 1.000 | 1.95 | 309 | 347 |
| tooljev-hosted-rerank | 0.985 | 1.000 | 0.985 | 1.000 | 1.000 | 1.06 | 299 | 420 |

### Abstention AUROC (in-catalog score, answerable vs not)

| benchmark | bm25 | dense-minilm | dense-mpnet | hybrid-bge | hybrid-rrf | tooljev-encoder | tooljev-encoder-hier | tooljev-encoder-rerank | tooljev-hosted-fit | tooljev-hosted-rerank |
|---|---|---|---|---|---|---|---|---|---|---|
| MCPToolBench++ (leave-category-out) | 0.751 | 0.913 | 0.915 | 0.803 | 0.769 | 0.721 | 0.814 | 0.712 | 0.897 | 0.894 |
| LiveMCPBench (leave-server-out) | 0.604 | 0.685 | 0.663 | 0.623 | 0.637 | 0.592 | 0.553 | 0.592 | 0.711 | 0.724 |
| When2Call | 0.628 | 0.735 | 0.726 | 0.531 | 0.534 | 0.884 | 0.869 | 0.901 | 0.967 | 0.942 |

### Threshold transfer (fraction handled correctly)

| router, threshold | t | bfcl-irrelevance abstain | livemcpbench keep | livemcpbench-lso abstain | mcptoolbench keep | mcptoolbench-lso abstain |
|---|---|---|---|---|---|---|
| bm25 @ fit on When2Call @90% TPR | 0.906 | 0.592 | 1.000 | 0.000 | 0.958 | 0.050 |
| dense-minilm @ fit on When2Call @90% TPR | 0.19 | 0.483 | 0.979 | 0.043 | 0.951 | 0.593 |
| dense-mpnet @ fit on When2Call @90% TPR | 0.188 | 0.492 | 0.979 | 0.064 | 0.969 | 0.360 |
| hybrid-rrf @ fit on When2Call @90% TPR | 0.0325 | 0.000 | 0.277 | 0.830 | 0.670 | 0.823 |
| tooljev-encoder-hier @ fit on When2Call @90% TPR | 0.032 | 0.654 | 0.979 | 0.021 | 0.981 | 0.063 |
| tooljev-encoder-hier @ default 0.5 | 0.5 | 0.783 | 0.372 | 0.702 | 0.873 | 0.427 |
| tooljev-encoder-rerank @ fit on When2Call @90% TPR | 0.014 | 0.673 | 0.894 | 0.128 | 0.987 | 0.107 |
| tooljev-encoder-rerank @ default 0.5 | 0.5 | 0.873 | 0.553 | 0.585 | 0.653 | 0.620 |
| tooljev-encoder @ fit on When2Call @90% TPR | 0.009 | 0.633 | 0.926 | 0.085 | 0.990 | 0.077 |
| tooljev-encoder @ default 0.5 | 0.5 | 0.858 | 0.553 | 0.585 | 0.664 | 0.610 |
| hybrid-bge @ fit on When2Call @90% TPR | 0.0489 | - | 0.330 | 0.798 | 0.701 | 0.873 |
| tooljev-hosted-fit @ fit on When2Call @90% TPR | 0.31 | - | 0.850 | 0.475 | 0.925 | 0.475 |
| tooljev-hosted-fit @ default 0.5 | 0.5 | - | 0.800 | 0.500 | 0.825 | 0.725 |
| tooljev-hosted-rerank @ fit on When2Call @90% TPR | 0.36 | - | 0.830 | 0.479 | 0.950 | 0.611 |
| tooljev-hosted-rerank @ default 0.5 | 0.5 | - | 0.787 | 0.543 | 0.845 | 0.749 |
