# Tutorial on Network Traffic Flow Classification Using Machine Learning

## Overview

This repository contains the official Jupyter notebooks that accompany our tutorial paper, **"Tutorial on Network Traffic Flow Classification Using Machine Learning."**

This work is designed to bridge the persistent gap between the theoretical principles of traffic classification and their practical, real-world implementation. The tutorial provides a complete, end-to-end blueprint for developing, evaluating, and understanding robust machine learning models for encrypted network traffic. The notebooks guide the user through a systematic process of discovery, deliberately exposing and addressing the common pitfalls and complex realities that researchers and practitioners encounter.

## Running the Notebooks

We recommend **downloading the notebooks** and running them in your preferred environment (e.g., locally with VS Code, Jupyter Lab, or by uploading them to Google Colab).

The notebooks are designed to be **self-contained and set up their own dependencies**. The initial cells in each notebook will automatically:
1.  Clone the required parts of this repository to fetch necessary data files.
2.  Install the exact Python package versions specified in the `requirements.txt` files.

This ensures that the environment is correctly configured with a single click, regardless of where you are running the notebook.

## A Note on Reproducibility

The text and numerical results within the notebooks were generated and validated in a specific environment (Google Colab with an x86-64 CPU, Python 3.12, using the package versions pinned in the `requirements.txt` files).

While we have taken every step to ensure reproducibility, you may observe numerical variations in some outputs (e.g., in the final decimal places of an F1-score). This is an expected and normal phenomenon in complex data science workflows and can be caused by:
*   **Different Hardware Architectures:** Low-level numerical libraries are optimized differently for CPUs from Intel (x86-64) versus Apple (ARM) or AMD.
*   **Different OS or Python Versions:** Subtle differences between operating systems or minor Python versions can lead to tiny variations in floating-point calculations.

**The Bottom Line:** These minor variations are normal and **do not change the overall conclusions or key lessons** of the tutorial. A model that is better will still be better, and the patterns you observe will be the same.

## Repository Structure and Notebooks Overview

The tutorial is organized into a series of modules, each in its own directory. The notebooks are designed to be run in sequence for a complete learning experience.

### 📂 `01-data-collection/`
This module is a masterclass in the principles and pitfalls of modern flow data collection using NFStream.

-   **`01-data-collection.ipynb`**: Teaches the complete process from raw packets to a feature-rich dataset. It covers foundational concepts (flow metering, statistical features, SPLT, nDPI labeling) and exposes critical, real-world challenges such as the **domain shift risk from NIC hardware offloading**, the "boundary flow" problem with split captures, and the power of extending the framework with **custom plugins**.

### 📂 `02-app-classification/`
This is the core of the tutorial, containing the data preparation and a "Three-Act" modeling journey.

-   **`02a-data-preparation.ipynb`**: Implements a rigorous, iterative pipeline for data cleaning and feature engineering. It demonstrates how to handle data leakage, justify the aggressive filtering of network "noise" (over 62% of raw flows), and strategically create the final datasets for modeling.

-   **`02b-comparative-modeling.ipynb`**: Executes the comparative "Three-Act" modeling strategy.
    -   **Act 1:** A model "bake-off" that proves the superiority of tree-based ensembles.
    -   **Act 2:** A head-to-head experiment that reveals the profound insight that **raw packet sequences (SPLT) can outperform** meticulously engineered statistical features.
    -   **Act 3:** A "real-world" challenge on 63 classes, teaching advanced, programmatic evaluation techniques.

-   **`02c-advanced-optimization.ipynb`**: A deep dive into the systematic workflow of model optimization. It guides the user from a simple baseline, through robust k-fold cross-validation and `GridSearchCV`, to the final, powerful conclusion that a **simpler, leaner model is superior** by removing noisy features.

### 📂 `03-explainability/`
This final module addresses the "black box" problem by teaching the practical application of eXplainable AI (XAI).

-   **`03-explainability.ipynb`**: Provides a hands-on guide to a spectrum of XAI techniques, from inherently interpretable models to advanced tools like **SHAP** and **LIME**. Its most critical lesson is in methodological rigor, proving how naive explainability methods can be misleading and teaching the correct way to handle common issues like feature correlation.

## Citation

This work is currently under review for publication. If you use the code or concepts from this tutorial in your research, we kindly ask that you cite our paper. We will update this section with the full citation and DOI upon publication.

In the meantime, you may use the following BibTeX entry for the preprint or unpublished work:

```bibtex
@unpublished{pekar2025tutorial,
  author    = {Adrián Pekár, Richard Plný, and Karel Hynek},
  title     = {Tutorial on Network Traffic Flow Classification Using Machine Learning},
  note      = {Submitted for publication},
  year      = {2025}
}
```

## Contact

For questions about the paper or the notebooks, please open an issue in this repository.

