#pragma once
#include "math.h"

// Transposée : Aᵀ
Table matrix_transpose(const Table *A);

// Produit matriciel : C = A · B  (A: n×m, B: m×p → C: n×p)
Table matrix_multiply(const Table *A, const Table *B);

// Inverse par Gauss-Jordan : A⁻¹
Table matrix_inverse(const Table *A);
