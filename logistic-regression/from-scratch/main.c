#include <common/math.h>
#include <common/matrix.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// fonction sigmoide
// σ(z) = 1 / (1 + e^(-z))
static inline f32 sigmoid(f32 z) { return 1.0f / (1.0f + exp(-z)); }

// Prediction des probabilites pour un ensemble d'echantillons
// X : matrice n×d (n echantillons, d caracteristiques)
// theta : vecteur colonne d×1 des parametres
// Retourne un vecteur colonne n×1 contenant σ(X · theta)
Table predict_probabilities(const Table *X, const Table *theta) {
  Table z = matrix_multiply(X, theta); // z = X * theta   (n×1)
  for (uint i = 0; i < z.rows; i++) {
    f32 val = table_get(&z, i, 0);
    table_set(&z, i, 0, sigmoid(val));
  }
  return z;
}

// Calcul de la perte (log‑vraisemblance negative / entropie croisee)
// J(θ) = -1/m * Σ [ y_i log(h_i) + (1-y_i) log(1-h_i) ]
// avec h_i = σ(x_i · θ)
f32 compute_loss(const Table *X, const Table *y, const Table *theta) {
  Table h = predict_probabilities(X, theta);
  uint m = X->rows;
  f32 loss = 0.0f;
  const f32 eps = 1e-7f; // pour eviter log(0)

  for (uint i = 0; i < m; i++) {
    f32 h_i = table_get(&h, i, 0);
    f32 y_i = table_get(y, i, 0);
    // Clamp pour stabilite numerique
    if (h_i < eps)
      h_i = eps;
    if (h_i > 1.0f - eps)
      h_i = 1.0f - eps;
    loss += y_i * logf(h_i) + (1.0f - y_i) * logf(1.0f - h_i);
  }
  free_table(&h);
  return -loss / (f32)m;
}

// Calcul du gradient de la perte par rapport a θ
// ∇J(θ) = (1/m) * Xᵀ · (h - y)   où h = σ(X·θ)
Table compute_gradient(const Table *X, const Table *y, const Table *theta) {
  Table h = predict_probabilities(X, theta);
  uint m = X->rows;
  uint d = X->cols;

  // erreur = h - y   (n×1)
  Table error = init_table(m, 1);
  for (uint i = 0; i < m; i++) {
    f32 diff = table_get(&h, i, 0) - table_get(y, i, 0);
    table_set(&error, i, 0, diff);
  }

  // Xᵀ a pour dimensions d×n
  Table X_T = init_table(d, m);
  for (uint i = 0; i < m; i++) {
    for (uint j = 0; j < d; j++) {
      table_set(&X_T, j, i, table_get(X, i, j));
    }
  }

  Table gradient = matrix_multiply(&X_T, &error); // d×1

  // Division par m
  for (uint i = 0; i < d; i++) {
    f32 val = table_get(&gradient, i, 0) / (f32)m;
    table_set(&gradient, i, 0, val);
  }

  free_table(&h);
  free_table(&error);
  free_table(&X_T);
  return gradient;
}

// Descente de gradient pour la regression logistique
// X : matrice des caracteristiques (n×d)
// y : vecteur des etiquettes (n×1)
// alpha : taux d'apprentissage
// nb_iterations : nombre d'iterations
// Retourne le vecteur θ appris (d×1)
Table gradient_descent(const Table *X, const Table *y, f32 alpha,
                       uint nb_iterations) {
  uint d = X->cols;
  Table theta = init_table(d, 1);

  // Initialisation des parametres a zero
  for (uint i = 0; i < d; i++) {
    table_set(&theta, i, 0, 0.0f);
  }

  printf("Debut de la descente de gradient...\n");
  for (uint iter = 0; iter < nb_iterations; iter++) {
    Table grad = compute_gradient(X, y, &theta);

    // Mise a jour : θ = θ - alpha * ∇J(θ)
    for (uint i = 0; i < d; i++) {
      f32 new_val = table_get(&theta, i, 0) - alpha * table_get(&grad, i, 0);
      table_set(&theta, i, 0, new_val);
    }

    // Affichage de la perte toutes les 100 iterations (ou a la fin)
    if (iter % 100 == 0 || iter == nb_iterations - 1) {
      f32 loss = compute_loss(X, y, &theta);
      printf("Iteration %4d | Perte = %.6f\n", iter, loss);
    }

    free_table(&grad);
  }

  return theta;
}

int main() {
  printf("=== Regression Logistique avec Descente de Gradient ===\n\n");

  // Donnees d'exemple : OU logique (OR)
  // X : 4 echantillons, 3 caracteristiques (une colonne de biais = 1)
  // y : etiquettes correspondantes
  const uint n = 4; // nombre d'echantillons
  const uint d = 3; // caracteristiques : [1, x1, x2]

  Table X = init_table(n, d);
  Table y = init_table(n, 1);

  // Remplissage manuel (biais = 1 pour chaque ligne)
  f32 X_data[4][3] = {
      {1.0f, 0.0f, 0.0f}, // (0,0) -> 0
      {1.0f, 0.0f, 1.0f}, // (0,1) -> 1
      {1.0f, 1.0f, 0.0f}, // (1,0) -> 1
      {1.0f, 1.0f, 1.0f}  // (1,1) -> 1
  };
  f32 y_data[4] = {0.0f, 1.0f, 1.0f, 1.0f};

  for (uint i = 0; i < n; i++) {
    for (uint j = 0; j < d; j++) {
      table_set(&X, i, j, X_data[i][j]);
    }
    table_set(&y, i, 0, y_data[i]);
  }

  printf("Donnees (probleme OU) :\n");
  for (uint i = 0; i < n; i++) {
    printf("x = (%.0f, %.0f) -> y = %.0f\n", table_get(&X, i, 1),
           table_get(&X, i, 2), table_get(&y, i, 0));
  }
  printf("\n");

  // Parametres de la descente de gradient
  f32 alpha = 0.5f;
  uint nb_iterations = 1000;

  // Apprentissage
  Table theta = gradient_descent(&X, &y, alpha, nb_iterations);

  // Affichage des parametres appris
  printf("\nParametres appris Theta :\n");
  printf("Theta 0 (biais) = %.4f\n", table_get(&theta, 0, 0));
  printf("Theta 1 (x1)    = %.4f\n", table_get(&theta, 1, 0));
  printf("Theta 2 (x2)    = %.4f\n", table_get(&theta, 2, 0));

  // Predictions finales
  Table probas = predict_probabilities(&X, &theta);
  printf("\nPredictions sur les donnees d'entrainement :\n");
  for (uint i = 0; i < n; i++) {
    f32 p = table_get(&probas, i, 0);
    printf("x = (%.0f, %.0f) -> p(y=1) = %.4f -> predit = %d\n",
           table_get(&X, i, 1), table_get(&X, i, 2), p, p >= 0.5f ? 1 : 0);
  }

  // Liberation de la memoire
  free_table(&X);
  free_table(&y);
  free_table(&theta);
  free_table(&probas);
  return 0;
}
