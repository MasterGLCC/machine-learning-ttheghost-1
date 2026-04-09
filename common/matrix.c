#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Transposee : on echange lignes et colonnes, res[j][i] = A[i][j]
Table matrix_transpose(const Table *A) {
  Table res = init_table(A->cols, A->rows);
  for (uint i = 0; i < A->rows; i++) {
    for (uint j = 0; j < A->cols; j++) {
      table_set(&res, j, i, table_get(A, i, j));
    }
  }
  return res;
}

// Produit matriciel : C = A · B
// Cᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ   (A: n×m, B: m×p → C: n×p)
Table matrix_multiply(const Table *A, const Table *B) {
  if (A->cols != B->rows) {
    fprintf(stderr, "Erreur dimensions produit matriciel\n");
    exit(EXIT_FAILURE);
  }
  Table C = init_table(A->rows, B->cols);
  const uint N = A->rows;
  const uint K = A->cols;
  const uint M = B->cols;

  for (uint i = 0; i < N; i++) {
    f32 *restrict C_row = C.data + i * M;
    const f32 *restrict A_row = A->data + i * K;
    for (uint k = 0; k < K; k++) {
      f32 a_ik = A_row[k];
      const f32 *restrict B_row = B->data + k * M;
      for (uint j = 0; j < M; j++) {
        C_row[j] += a_ik * B_row[j];
      }
    }
  }
  return C;
}

// Inversion par elimination de Gauss-Jordan avec pivot partiel
// On construit la matrice augmentee [A | I] puis on reduit pour obtenir [I |
// A⁻¹]
Table matrix_inverse(const Table *A) {
  if (A->rows != A->cols) {
    fprintf(stderr, "L'inverse n'existe que pour les matrices carrees\n");
    exit(EXIT_FAILURE);
  }
  uint n = A->rows;
  uint n2 = 2 * n;

  // Matrice augmentee [A | I] de taille n × 2n
  Table aug = init_table(n, 2 * n);
  f32 *restrict aug_data = aug.data;
  const f32 *restrict A_data = A->data;

  // Remplissage de la matrice augmentee : A a gauche, I a droite
  for (uint i = 0; i < n; i++) {
    f32 *row = aug_data + i * n2;
    const f32 *A_row = A_data + i * n;
    // Copier la partie A de la matrice augmentee
    for (uint j = 0; j < n; j++)
      row[j] = A_row[j];
    // la partie identite a droite
    for (uint j = n; j < n2; j++)
      row[j] = 0.0f;
    row[n + i] = 1.0f;
  }

  // elimination de Gauss-Jordan
  for (uint col = 0; col < n; col++) {
    // Recherche du pivot partiel (plus grande valeur absolue dans la colonne)
    uint pivot = col;
      f32 max_val = fabsf(aug_data[col * n2 + col]);
    for (uint row = col + 1; row < n; row++) {
      f32 val = fabsf(aug_data[row * n2 + col]);
      if (val > max_val) {
        max_val = val;
        pivot = row;
      }
    }
    if (max_val < 1e-8f) {
      fprintf(stderr, "Matrice singulière, pas d'inverse\n");
      exit(EXIT_FAILURE);
    }

    // echange de lignes si le pivot n'est pas sur la diagonale
    if (pivot != col) {
      f32 *row_col = aug_data + col * n2;
      f32 *row_piv = aug_data + pivot * n2;
      for (uint j = 0; j < n2; j++) {
        f32 tmp = row_col[j];
        row_col[j] = row_piv[j];
        row_piv[j] = tmp;
      }
    }

    // On divise la ligne pivot pour que l'element diagonal = 1
    f32 *pivot_row = aug_data + col * n2;
    f32 inv_pivot = 1.0f / pivot_row[col];
    for (uint j = 0; j < 2 * n; j++) {
      pivot_row[j] *= inv_pivot;
    }

    // elimination sur les autres lignes
    for (uint row = 0; row < n; row++) {
      if (row == col)
        continue;
      f32 factor = aug_data[row * n2 + col];
      if (fabsf(factor) < 1e-12f)
        continue;
      
      f32 *target_row = aug_data + row * n2;
      for (uint j = 0; j < n2; j++)
        target_row[j] -= factor * pivot_row[j];
    }
  }

  // On recupère A⁻¹ dans la partie droite de la matrice augmentee
  Table inv = init_table(n, n);
  f32 *restrict inv_data = inv.data;
  for (uint i = 0; i < n; i++) {
    f32 *src = aug_data + i * n2 + n;
    f32 *dst = inv_data + i * n;
    for (uint j = 0; j < n; j++)
      dst[j] = src[j];
  }

  free_table(&aug);
  return inv;
}
