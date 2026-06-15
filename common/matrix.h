#pragma once
#include "math.h"

// Addition matricielle element par element : C_ij = A_ij + B_ij
Table table_add(const Table *A, const Table *B);

// Soustraction matricielle element par element : C_ij = A_ij - B_ij
Table table_sub(const Table *A, const Table *B);

// Division de tous les elements d'une table par un scalaire : C_ij = A_ij / s
Table table_div_scalar(const Table *A, f32 scalar);

// Transposee d'une matrice : B = Aᵀ ou B_ji = A_ij
Table matrix_transpose(const Table *A);

// Produit matriciel : C = A · B de dimension n x p (avec A de dim n x m et B de dim m x p)
// Formule : C_ik = ∑ (A_ij * B_jk) pour j allant de 0 a m-1
Table matrix_multiply(const Table *A, const Table *B);

// Inverse d'une matrice par l'elimination de Gauss-Jordan : B = A⁻¹ tel que A · B = I
Table matrix_inverse(const Table *A);
