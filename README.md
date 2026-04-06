# ML Assignment No 1

[![Build Status](https://github.com/MasterGLCC/machine-learning-ttheghost-1/actions/workflows/makefile.yml/badge.svg)](https://github.com/MasterGLCC/machine-learning-ttheghost-1/actions)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jyj5d-5u)

Ce dépôt contient les implémentations pour la régression linéaire, multiple et polynomiale.

Chaque méthode de régression est implémentée de deux manières différentes :

1. Une version "from scratch" codée entièrement en C.
2. Une version utilisant des bibliothèques standards (Python, scikit-learn, PyTorch).

## La structure du repo

Le repo est divisé en 5 dossiers principaux :

- `datasets` : ce dossier contient les fichiers `.csv` utilisés pour entraîner et tester les modèles.
- `linear-regression` : implémentations de la régression linéaire simple.
- `multiple-linear-regression` : implémentations de la régression linéaire multiple.
- `poly-linear-regression` : implémentations de la régression polynomiale.
- `common` : ce dossier contient les fichiers `.c` et `.h` utilisés en commun par les implémentations from scratch.

Chaque dossier de régression contient lui-même deux sous-dossiers : `from-scratch` et `with-library`.
