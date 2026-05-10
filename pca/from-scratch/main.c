/**
 * Mohammed IFKIRNE
 */

/**
 * Implementation de l'Analyse en Composantes Principales (PCA)
 *
 * L'ACP (ou PCA en anglais) est un algorithme d'apprentissage non-supervise
 * utilise pour la reduction de dimensionnalite. Il projette les donnees
 * de haute dimension vers une dimension inferieure tout en maximisant la
 * variance conservee (minimiser la perte d'information).
 *
 * Concepts mathematiques cles:
 * 1. Standardisation : Centrer (moyenne=0) et reduire (ecart-type=1) chaque
 * variable. x_std = (x - mu) / sigma
 *
 * 2. Matrice de Covariance : Calcul de la matrice Sigma = (X^T * X) / (n - 1)
 *    Elle capture les correlations entre toutes les paires de caracteristiques.
 *
 * 3. Vecteurs Propres et Valeurs Propres : Les directions de plus grande
 * variance (composantes principales) sont les vecteurs propres (Eigenvectors)
 * de Sigma. Ici on utilise la "Methode de la Puissance Iteree" avec Deflation
 * pour trouver les 'k' vecteurs propres dominants de maniere numerique.
 *
 * 4. Projection : Les donnees d'origine X (n x d) sont multipliees par la
 *    matrice de projection W (d x k) pour obtenir les donnees reduites Z (n x
 * k). Z = X * W
 */

#include "common/csv.h"
#include "common/math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define K_COMPONENTS 2 // On reduit a 2 dimensions
#define ITERATIONS                                                             \
  100 // Nombre d'iterations pour la methode de la puissance iteree

// 1. Fonction pour standardiser une table (Z-Score)
void standardize(Table *X) {
  for (uint j = 0; j < X->cols; j++) {
    // Calcul de la moyenne
    f32 sum = 0.0f;
    for (uint i = 0; i < X->rows; i++) {
      sum += table_get(X, i, j);
    }
    f32 mean = sum / X->rows;

    // Calcul de l'ecart-type
    f32 sum_sq = 0.0f;
    for (uint i = 0; i < X->rows; i++) {
      f32 diff = table_get(X, i, j) - mean;
      sum_sq += diff * diff;
    }
    f32 std_dev = sqrtf(sum_sq / (X->rows - 1));
    if (std_dev == 0.0f)
      std_dev = 1e-6f; // Securite contre division par 0

    // Application de la standardisation
    for (uint i = 0; i < X->rows; i++) {
      f32 z = (table_get(X, i, j) - mean) / std_dev;
      table_set(X, i, j, z);
    }
  }
}

// 2. Calcule la matrice de covariance: Cov = (X^T * X) / (n - 1)
f32 **compute_covariance(const Table *X) {
  f32 **cov = malloc(X->cols * sizeof(f32 *));
  for (uint i = 0; i < X->cols; i++) {
    cov[i] = calloc(X->cols, sizeof(f32));
  }

  for (uint j1 = 0; j1 < X->cols; j1++) {
    for (uint j2 = 0; j2 < X->cols; j2++) {
      f32 sum = 0.0f;
      for (uint i = 0; i < X->rows; i++) {
        sum += table_get(X, i, j1) * table_get(X, i, j2);
      }
      cov[j1][j2] = sum / (X->rows - 1);
    }
  }
  return cov;
}

// 3. Methode de la Puissance Iteree pour trouver le vecteur propre dominant
void power_iteration(f32 **matrix, uint n, f32 *eigenvector, f32 *eigenvalue) {
  // Initialisation aleatoire du vecteur
  for (uint i = 0; i < n; i++) {
    eigenvector[i] = ((f32)rand() / RAND_MAX);
  }

  f32 *temp = malloc(n * sizeof(f32));

  for (int iter = 0; iter < ITERATIONS; iter++) {
    // temp = Matrix * eigenvector
    for (uint i = 0; i < n; i++) {
      temp[i] = 0.0f;
      for (uint j = 0; j < n; j++) {
        temp[i] += matrix[i][j] * eigenvector[j];
      }
    }

    // Normalisation de temp
    f32 norm = 0.0f;
    for (uint i = 0; i < n; i++) {
      norm += temp[i] * temp[i];
    }
    norm = sqrtf(norm);

    for (uint i = 0; i < n; i++) {
      eigenvector[i] = temp[i] / norm;
    }
  }

  // Calcul de la valeur propre (Rayleigh Quotient) : lambda = v^T * M * v
  *eigenvalue = 0.0f;
  for (uint i = 0; i < n; i++) {
    temp[i] = 0.0f;
    for (uint j = 0; j < n; j++) {
      temp[i] += matrix[i][j] * eigenvector[j];
    }
    *eigenvalue += eigenvector[i] * temp[i];
  }

  free(temp);
}

// Deflation : on retire l'influence du vecteur propre dominant trouve de la
// matrice M' = M - lambda * (v * v^T)
void deflate_matrix(f32 **matrix, uint n, const f32 *eigenvector,
                    f32 eigenvalue) {
  for (uint i = 0; i < n; i++) {
    for (uint j = 0; j < n; j++) {
      matrix[i][j] -= eigenvalue * (eigenvector[i] * eigenvector[j]);
    }
  }
}

int main() {
  srand((unsigned int)time(NULL));

  printf("--- PCA (Analyse en Composantes Principales) ---\n");

  Table X = table_load_csv("datasets/pca_data.csv", 1);
  printf("Dataset charge : %u echantillons, %u dimensions (features).\n",
         X.rows, X.cols);

  if (X.cols < K_COMPONENTS) {
    printf(
        "Erreur : Impossible de reduire a %d dimensions car X n'en a que %u.\n",
        K_COMPONENTS, X.cols);
    return 1;
  }

  // 1. Standardisation
  standardize(&X);
  printf("Standardisation Z-Score (Moyenne=0, Ecart-type=1) effectuee.\n");

  // 2. Matrice de Covariance
  f32 **cov = compute_covariance(&X);

  // Calcul de la variance totale pour le ratio de variance expliquee
  f32 total_variance = 0.0f;
  for (uint i = 0; i < X.cols; i++) {
    total_variance +=
        cov[i][i]; // La trace de la matrice = Somme des valeurs propres
  }

  // 3. Extraction des Composantes Principales
  f32 **eigenvectors = malloc(K_COMPONENTS * sizeof(f32 *));
  f32 *eigenvalues = malloc(K_COMPONENTS * sizeof(f32));

  printf("\nRecherche des %d composantes principales...\n", K_COMPONENTS);

  for (uint k = 0; k < K_COMPONENTS; k++) {
    eigenvectors[k] = malloc(X.cols * sizeof(f32));

    // Trouver la composante dominante actuelle
    power_iteration(cov, X.cols, eigenvectors[k], &eigenvalues[k]);

    // Retirer cette composante pour trouver la suivante
    deflate_matrix(cov, X.cols, eigenvectors[k], eigenvalues[k]);

    f32 explained_var = (eigenvalues[k] / total_variance) * 100.0f;
    printf(
        "Composante %d : Valeur propre = %.4f (Variance expliquee = %5.2f%%)\n",
        k + 1, eigenvalues[k], explained_var);
  }

  // 4. Projection des donnees : Z = X * W
  printf("\nProjection des 5 premiers echantillons de 3D vers 2D :\n");
  for (uint i = 0; i < 5 && i < X.rows; i++) {
    printf("Echantillon %u : [ ", i);
    for (uint k = 0; k < K_COMPONENTS; k++) {
      f32 proj = 0.0f;
      for (uint j = 0; j < X.cols; j++) {
        proj += table_get(&X, i, j) * eigenvectors[k][j];
      }
      printf("%7.4f ", proj);
    }
    printf("]\n");
  }

  // Nettoyage de la memoire
  for (uint i = 0; i < X.cols; i++)
    free(cov[i]);
  free(cov);

  for (uint k = 0; k < K_COMPONENTS; k++)
    free(eigenvectors[k]);
  free(eigenvectors);
  free(eigenvalues);
  free_table(&X);

  return 0;
}

// Affichage

// --- PCA (Analyse en Composantes Principales) ---
// Charge 'datasets/pca_data.csv': 10 lignes x 3 colonnes
// Dataset charge : 10 echantillons, 3 dimensions (features).
// Standardisation Z-Score (Moyenne=0, Ecart-type=1) effectuee.

// Recherche des 2 composantes principales...
// Composante 1 : Valeur propre = 2.9259 (Variance expliquee = 97.53%)
// Composante 2 : Valeur propre = 0.0741 (Variance expliquee =  2.47%)

// Projection des 5 premiers echantillons de 3D vers 2D :
// Echantillon 0 : [  1.2669  0.2126 ]
// Echantillon 1 : [ -2.6971 -0.1701 ]
// Echantillon 2 : [  1.4599 -0.4753 ]
// Echantillon 3 : [  0.4011 -0.1611 ]
// Echantillon 4 : [  2.5500  0.2523 ]