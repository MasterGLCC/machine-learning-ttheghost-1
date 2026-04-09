# ML Assignment No 1

[![Build Status](https://github.com/MasterGLCC/machine-learning-ttheghost-1/actions/workflows/makefile.yml/badge.svg)](https://github.com/MasterGLCC/machine-learning-ttheghost-1/actions)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jyj5d-5u)

Ce depot contient les implementations pour la regression lineaire, multiple et polynomiale.

Chaque methode de regression est implementee de deux manieres :

1. Une version "from scratch" codee en C.
2. Une version avec bibliotheques Python (scikit-learn / PyTorch) dans des notebooks Jupyter.

## Structure du repo

```bash
.
├── common/              # code C partage (table, matrice, csv, stats)
│   ├── math.h / math.c         # structure Table, moyenne, ecart-type, z-score
│   ├── matrix.h / matrix.c     # transposee, produit, inverse (Gauss-Jordan)
│   └── csv.h / csv.c           # chargement de fichiers CSV
├── datasets/
│   └── SOCR-HeightWeight.csv
├── linear-regression/
│   ├── from-scratch/main.c                # descente de gradient univariee
│   └── with-lib/                          # notebooks jupyter
├── multiple-linear-regression/
│   ├── from-scratch/main.c                # equation normale (XᵀX)⁻¹Xᵀy
│   └── with-lib/
├── poly-linear-regression/
│   ├── from-scratch/main.c                # regression polynomiale degre 2
│   └── with-lib/
└── Makefile
```

## Compilation et execution

Prerequis : `clang` et `make`.

```bash
# compiler tout
make

# executer
make run_lr     # regression lineaire
make run_mlr    # regression multiple
make run_plr    # regression polynomiale

# nettoyer
make clean
```
