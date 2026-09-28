
### MCPToolBench++ routing (n=211)

| router | top1 | recall@5 | server@1 | category@1 | top1@50%cov | shortlist_hit | shortlist_size | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.564 | 0.763 | 0.668 | 0.735 | 0.810 | - | - | 0 | 0 |
| dense-minilm | 0.607 | 0.825 | 0.744 | 0.919 | 0.752 | - | - | 5 | 10 |
| dense-mpnet | 0.630 | 0.853 | 0.720 | 0.919 | 0.781 | - | - | 10 | 16 |
| hybrid-rrf | 0.630 | 0.839 | 0.735 | 0.844 | 0.886 | - | - | 5 | 8 |
| tooljev-encoder-hier | 0.156 | 0.327 | 0.303 | 0.531 | 0.238 | 0.322 | 4.76 | 719 | 1408 |
| tooljev-encoder-rerank | 0.193 | 0.587 | 0.447 | 0.627 | 0.227 | 0.567 | 4.77 | 162 | 246 |
| tooljev-encoder | 0.630 | 0.839 | 0.735 | 0.844 | 0.638 | 0.564 | 2.82 | 74 | 96 |
| hybrid-bge | 0.725 | 0.877 | 0.763 | 0.948 | 0.981 | - | - | 8 | 10 |
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

### When2Call routing (n=201)

| router | top1 | recall@5 | server@1 | top1@50%cov | shortlist_hit | shortlist_size | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.721 | 1.000 | 0.721 | 0.750 | - | - | 0 | 0 |
| dense-minilm | 0.886 | 1.000 | 0.886 | 0.900 | - | - | 3 | 5 |
| dense-mpnet | 0.900 | 1.000 | 0.900 | 0.910 | - | - | 11 | 23 |
| hybrid-rrf | 0.796 | 1.000 | 0.796 | 1.000 | - | - | 4 | 9 |
| tooljev-encoder-hier | 0.667 | 0.980 | 0.667 | 0.880 | 0.945 | 1.78 | 55 | 241 |
| tooljev-encoder-rerank | 0.667 | 1.000 | 0.667 | 0.893 | 0.980 | 2.19 | 40 | 114 |
| tooljev-encoder | 0.796 | 1.000 | 0.796 | 0.950 | 0.950 | 1.74 | 28 | 78 |
| hybrid-bge | 0.915 | 1.000 | 0.915 | 1.000 | - | - | 8 | 10 |
| tooljev-hosted-fit | 0.900 | 1.000 | 0.900 | 1.000 | 1.000 | 1.95 | 309 | 347 |
| tooljev-hosted-rerank | 0.985 | 1.000 | 0.985 | 1.000 | 1.000 | 1.06 | 299 | 420 |

### Abstention AUROC (in-catalog score, answerable vs not)

| benchmark | bm25 | dense-minilm | dense-mpnet | hybrid-bge | hybrid-rrf | tooljev-encoder | tooljev-encoder-hier | tooljev-encoder-rerank | tooljev-hosted-fit | tooljev-hosted-rerank |
|---|---|---|---|---|---|---|---|---|---|---|
| MCPToolBench++ (leave-category-out) | 0.772 | 0.906 | 0.907 | 0.802 | 0.768 | 0.688 | 0.785 | 0.696 | 0.897 | 0.894 |
| LiveMCPBench (leave-server-out) | 0.604 | 0.685 | 0.663 | 0.623 | 0.637 | 0.592 | 0.553 | 0.592 | 0.711 | 0.724 |
| When2Call | 0.611 | 0.739 | 0.725 | 0.534 | 0.534 | 0.890 | 0.876 | 0.909 | 0.967 | 0.942 |

### Threshold transfer (fraction handled correctly)

| router, threshold | t | livemcpbench keep | livemcpbench-lso abstain | mcptoolbench keep | mcptoolbench-lso abstain |
|---|---|---|---|---|---|
| bm25 @ fit on When2Call @90% TPR | 1.05 | 1.000 | 0.000 | 0.972 | 0.038 |
| dense-minilm @ fit on When2Call @90% TPR | 0.19 | 0.979 | 0.043 | 0.938 | 0.587 |
| dense-mpnet @ fit on When2Call @90% TPR | 0.188 | 0.979 | 0.064 | 0.938 | 0.333 |
| hybrid-rrf @ fit on When2Call @90% TPR | 0.0325 | 0.277 | 0.830 | 0.701 | 0.798 |
| tooljev-encoder-hier @ fit on When2Call @90% TPR | 0.049 | 0.979 | 0.021 | 0.957 | 0.160 |
| tooljev-encoder-hier @ default 0.5 | 0.5 | 0.372 | 0.702 | 0.834 | 0.423 |
| tooljev-encoder-rerank @ fit on When2Call @90% TPR | 0.014 | 0.894 | 0.128 | 0.987 | 0.091 |
| tooljev-encoder-rerank @ default 0.5 | 0.5 | 0.553 | 0.585 | 0.653 | 0.600 |
| tooljev-encoder @ fit on When2Call @90% TPR | 0.014 | 0.894 | 0.128 | 0.986 | 0.089 |
| tooljev-encoder @ default 0.5 | 0.5 | 0.553 | 0.585 | 0.649 | 0.592 |
| hybrid-bge @ fit on When2Call @90% TPR | 0.0489 | 0.330 | 0.798 | 0.701 | 0.873 |
| tooljev-hosted-fit @ fit on When2Call @90% TPR | 0.31 | 0.850 | 0.475 | 0.925 | 0.475 |
| tooljev-hosted-fit @ default 0.5 | 0.5 | 0.800 | 0.500 | 0.825 | 0.725 |
| tooljev-hosted-rerank @ fit on When2Call @90% TPR | 0.36 | 0.830 | 0.479 | 0.950 | 0.611 |
| tooljev-hosted-rerank @ default 0.5 | 0.5 | 0.787 | 0.543 | 0.845 | 0.749 |
