#include "common/csv.h"
#include "common/math.h"
#include <assert.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

// Régression linéaire simple : y = a·x + b
typedef struct {
  f32 a;
  f32 b;
} LinearRegression;

// Prédiction pour une seule valeur : ŷ = a·x + b
inline f32 predict(const LinearRegression *model, f32 x) {
  return model->a * x + model->b;
}

// Prédiction sur tout le dataset (colonne par colonne)
inline Table batch_predict(const LinearRegression *model, const Table *X) {
  Table res = init_table(X->rows, 1);
  for (uint i = 0; i < X->rows; i++) {
    table_set(&res, i, 0, predict(model, table_get(X, i, 0)));
  }
  return res;
}

// Descente de gradient pour une variable
// On minimise la MSE : J(a,b) = (1/2n) Σᵢ (ŷᵢ - yᵢ)²
// Gradients :
//   ∂J/∂a = (1/n) Σᵢ (ŷᵢ - yᵢ) · xᵢ
//   ∂J/∂b = (1/n) Σᵢ (ŷᵢ - yᵢ)
void univariate_gradient_descent(LinearRegression *model, const Table *X,
                                 const Table *Y, f32 lr, uint max_iters) {
  assert(X->cols == 1 && Y->cols == 1);
  assert(X->rows == Y->rows);

  f32 last_loss = INFINITY;
  for (uint iter = 0; iter < max_iters; iter++) {
    f32 sum_a = 0.0;
    f32 sum_b = 0.0;
    f32 mse = 0.0;

    for (uint i = 0; i < X->rows; i++) {
      f32 x = table_get(X, i, 0);
      f32 y = table_get(Y, i, 0);
      f32 error = predict(model, x) - y;  // erreur = ŷ - y
      sum_a += error * x;
      sum_b += error;
      mse += error * error;
    }

    // Gradients moyennés sur n échantillons
    f32 dA = sum_a / X->rows;
    f32 dB = sum_b / X->rows;
    f32 current_loss = (1.0 / (2.0 * X->rows)) * mse;

    // Mise à jour des paramètres : θ = θ - α · ∂J/∂θ
    model->a -= lr * dA;
    model->b -= lr * dB;

    if (iter % 100 == 0) {
      printf("Iter: %u, Erreur: %f\n", iter, current_loss);
    }
    // Critère d'arrêt : convergence si la perte ne bouge presque plus
    if (fabsf(last_loss - current_loss) < 1e-7f) {
      printf("Convergence à l'itération %u\n", iter);
      break;
    }
    if (isnan(current_loss) || isinf(current_loss)) {
        printf("Erreur numérique détectée\n");
    }
    last_loss = current_loss;
  }
}

int main() {
  // Chargement du dataset taille/poids
  Table c = table_load_csv("datasets/SOCR-HeightWeight.csv", 1);

  // Normalisation z-score pour stabiliser la descente de gradient
  Table mean = table_mean_axis0(&c);
  Table stddev = table_stddev_axis0(&c, &mean);
  table_normlize_zscore_axis0(&c, &mean, &stddev);
  Table X = table_extract_column(&c, 1);
  Table Y = table_extract_column(&c, 2);

  LinearRegression model = {0, 0};

  univariate_gradient_descent(&model, &X, &Y, 0.1, 3500);

  f32 mean_x = table_get(&mean, 0, 1);
  f32 mean_y = table_get(&mean, 0, 2);
  f32 stddev_x = table_get(&stddev, 0, 1);
  f32 stddev_y = table_get(&stddev, 0, 2);

  // Reconversion des paramètres normalisés vers l'échelle d'origine
  // a_orig = (σ_y / σ_x) · a_norm
  // b_orig = μ_y - a_orig · μ_x + σ_y · b_norm
  f32 a_orig = (stddev_y / stddev_x) * model.a;
  f32 b_orig = mean_y - a_orig * mean_x + stddev_y * model.b;

  printf("Normalisé: a = %f, b = %f\n", model.a, model.b);
  printf("Échelle originale: a = %f, b = %f\n", a_orig, b_orig);

  free_table(&X);
  free_table(&Y);
  free_table(&mean);
  free_table(&stddev);
  free_table(&c);
  return 0;
};