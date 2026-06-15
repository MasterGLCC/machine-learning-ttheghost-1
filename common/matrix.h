#pragma once
#include "math.h"

// Additions
Table table_add(const Table *A, const Table *B);
Table table_sub(const Table *A, const Table *B);
Table table_div_scalar(const Table *A, f32 scalar);

// Transposee : Aᵀ
Table matrix_transpose(const Table *A);

// Produit matriciel : C = A · B  (A: n×m, B: m×p → C: n×p)
Table matrix_multiply(const Table *A, const Table *B);

// Inverse par Gauss-Jordan : A⁻¹
Table matrix_inverse(const Table *A);
