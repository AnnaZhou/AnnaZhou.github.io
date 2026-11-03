# Learning Through Information Forces

A GitHub Pages–ready, long-form research tutorial about kernel information potentials, Rényi entropy, information forces, divergence optimization, and quadratic mutual information, with **figures regenerated from the accompanying executed notebook**.

## Contents
- `index.html` — complete responsive article, equations (MathJax), figure zoom, interactive bandwidth slider, sidebar navigation
- `assets/` — all 11 experimental figures from the executed notebook
- `ITL_executed.ipynb` — fully executed source notebook with outputs

## Publish on GitHub Pages
1. Create a public GitHub repository, e.g. `itl-visual-experiments`.
2. Upload `index.html`, the `assets` folder, and `ITL_executed.ipynb` to the repository root.
3. Go to **Settings → Pages**, choose **Deploy from a branch**, select `main` and `/ (root)`, and save.
4. Visit `https://YOUR_USERNAME.github.io/itl-visual-experiments/` after deployment.

For a personal GitHub Pages repository named `YOUR_USERNAME.github.io`, the site URL is `https://YOUR_USERNAME.github.io/`.

To preview locally, run `python -m http.server 8000` in this folder, then open `http://localhost:8000/`.

## Dependencies
The webpage needs no build process. It uses Google's hosted font CSS and MathJax CDN for mathematical typesetting; the images and page content are local. The accompanying notebook uses NumPy, Matplotlib, and PyTorch.

## Attribution
Based on the mathematical framework in José C. Principe, *Information Theoretic Learning: Rényi's Entropy and Kernel Perspectives* (Springer, 2010), DOI: 10.1007/978-1-4419-1570-2. All experiment plots are generated from the accompanying synthetic-data notebook, not reproduced from the book.

Layout inspiration: https://rikiyat.github.io/blog/weight-space-programming/ . The design and implementation are original.
