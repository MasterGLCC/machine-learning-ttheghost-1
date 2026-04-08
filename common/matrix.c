#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Transposée : on échange lignes et colonnes, res[j][i] = A[i][j]
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
  for (uint i = 0; i < A->rows; i++) {
    for (uint j = 0; j < B->cols; j++) {
      f32 sum = 0.0f;
      for (uint k = 0; k < A->cols; k++) {
        sum += table_get(A, i, k) * table_get(B, k, j);
      }
      table_set(&C, i, j, sum);
    }
  }
  return C;
}

// Inversion par élimination de Gauss-Jordan avec pivot partiel
// On construit la matrice augmentée [A | I] puis on réduit pour obtenir [I |
// A⁻¹]
Table matrix_inverse(const Table *A) {
  if (A->rows != A->cols) {
    fprintf(stderr, "L'inverse n'existe que pour les matrices carrées\n");
    exit(EXIT_FAILURE);
  }
  uint n = A->rows;

  // Matrice augmentée [A | I] de taille n × 2n
  Table aug = init_table(n, 2 * n);
  for (uint i = 0; i < n; i++) {
    for (uint j = 0; j < n; j++) {
      table_set(&aug, i, j, table_get(A, i, j));
    }
    table_set(&aug, i, n + i, 1.0f); // identité à droite
  }

  // Élimination de Gauss-Jordan
  for (uint col = 0; col < n; col++) {
    // Recherche du pivot partiel (plus grande valeur absolue dans la colonne)
    uint pivot = col;
    f32 max_val = fabsf(table_get(&aug, col, col));
    for (uint row = col + 1; row < n; row++) {
      f32 val = fabsf(table_get(&aug, row, col));
      if (val > max_val) {
        max_val = val;
        pivot = row;
      }
    }
    if (max_val < 1e-8f) {
      fprintf(stderr, "Matrice singulière, pas d'inverse\n");
      exit(EXIT_FAILURE);
    }

    // Échange de lignes si le pivot n'est pas sur la diagonale
    if (pivot != col) {
      for (uint j = 0; j < 2 * n; j++) {
        f32 tmp = table_get(&aug, col, j);
        table_set(&aug, col, j, table_get(&aug, pivot, j));
        table_set(&aug, pivot, j, tmp);
      }
    }

    // On divise la ligne pivot pour que l'élément diagonal = 1
    f32 pivot_val = table_get(&aug, col, col);
    for (uint j = 0; j < 2 * n; j++) {
      table_set(&aug, col, j, table_get(&aug, col, j) / pivot_val);
    }

    // Élimination sur les autres lignes
    for (uint row = 0; row < n; row++) {
      if (row != col) {
        f32 factor = table_get(&aug, row, col);
        if (fabsf(factor) < 1e-12f)
          continue;
        for (uint j = 0; j < 2 * n; j++) {
          f32 new_val =
              table_get(&aug, row, j) - factor * table_get(&aug, col, j);
          table_set(&aug, row, j, new_val);
        }
      }
    }
  }

  // On récupère A⁻¹ dans la partie droite de la matrice augmentée
  Table inv = init_table(n, n);
  for (uint i = 0; i < n; i++) {
    for (uint j = 0; j < n; j++) {
      table_set(&inv, i, j, table_get(&aug, i, n + j));
    }
  }
  free_table(&aug);
  return inv;
}
