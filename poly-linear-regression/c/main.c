#include <common/math.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

// Regression polynomiale de degre 2 : ŷ = w₀ + w₁·x + w₂·x²
typedef struct {
  f32 w0, w1, w2;
} PolyRegression;

// Construction de la matrice de design polynomiale (degre 2)
// On normalise x d'abord : x' = (x - μ) / σ
// Colonnes resultantes : [1, x', x'²]
Table build_poly2_design(const Table *X, f32 *mean_x, f32 *std_x) {
  Table X_poly = init_table(X->rows, 3);

  // Calcul de la moyenne et de l'ecart-type de x
  f32 sum = 0.0f, sum_sq = 0.0f;
  for (uint i = 0; i < X->rows; i++) {
    f32 x = table_get(X, i, 0);
    sum += x;
    sum_sq += x * x;
  }
  *mean_x = sum / X->rows;
  // σ = √( E[x²] - (E[x])² )
  *std_x = sqrtf(sum_sq / X->rows - (*mean_x) * (*mean_x));
  if (*std_x < 1e-8f)
    *std_x = 1.0f;

  // Remplissage : colonne 0 = 1 (biais), colonne 1 = x', colonne 2 = x'²
  for (uint i = 0; i < X->rows; i++) {
    f32 x_norm = (table_get(X, i, 0) - *mean_x) / *std_x;
    table_set(&X_poly, i, 0, 1.0f);
    table_set(&X_poly, i, 1, x_norm);
    table_set(&X_poly, i, 2, x_norm * x_norm);
  }
  return X_poly;
}

// Descente de gradient pour la regression polynomiale
// On minimise J(w) = (1/2n) Σᵢ (ŷᵢ - yᵢ)²
// Gradients : ∂J/∂wⱼ = (1/n) Σᵢ (ŷᵢ - yᵢ) · xᵢⱼ
void poly_gradient_descent(PolyRegression *model, const Table *X_poly,
                           const Table *Y, f32 lr, uint max_iters) {
  uint n = X_poly->rows;
  f32 last_loss = INFINITY;

  for (uint iter = 0; iter < max_iters; iter++) {
    f32 grad0 = 0.0f, grad1 = 0.0f, grad2 = 0.0f;
    f32 loss = 0.0f;

    for (uint i = 0; i < n; i++) {
      f32 x0 = table_get(X_poly, i, 0); // toujours 1
      f32 x1 = table_get(X_poly, i, 1); // x normalise
      f32 x2 = table_get(X_poly, i, 2); // x² normalise
      f32 y = table_get(Y, i, 0);

      // ŷ = w₀·1 + w₁·x' + w₂·x'²
      f32 pred = model->w0 * x0 + model->w1 * x1 + model->w2 * x2;
      f32 error = pred - y;

      grad0 += error * x0;
      grad1 += error * x1;
      grad2 += error * x2;
      loss += error * error;
    }

    // Moyenne des gradients
    grad0 /= n;
    grad1 /= n;
    grad2 /= n;
    loss /= (2.0f * n);

    // Mise a jour : wⱼ = wⱼ - α · ∂J/∂wⱼ
    model->w0 -= lr * grad0;
    model->w1 -= lr * grad1;
    model->w2 -= lr * grad2;

    if (iter % 100 == 0) {
      printf("Iter %4u: loss = %f, w0=%f, w1=%f, w2=%f\n", iter, loss,
             model->w0, model->w1, model->w2);
    }

    // Convergence si la perte varie moins que 1e-6f
    if (fabsf(last_loss - loss) < 1e-6f) {
      printf("Convergence a l'iteration %u\n", iter);
      break;
    }
    last_loss = loss;
  }
}

// Prediction pour une valeur x dans l'espace original
// On renormalise avant de calculer ŷ
f32 poly_predict(const PolyRegression *model, f32 x_orig, f32 mean_x,
                 f32 std_x) {
  f32 x_norm = (x_orig - mean_x) / std_x;
  return model->w0 + model->w1 * x_norm + model->w2 * (x_norm * x_norm);
}

int main() {
  // Donnees quadratiques : y = 2 + 3x - 0.5x²
  uint n_samples = 7;
  Table X = init_table(n_samples, 1);
  Table Y = init_table(n_samples, 1);

  f32 x_vals[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  f32 y_vals[] = {
      2 + 3 * (-2) - 0.5 * 4, 2 + 3 * (-1) - 0.5 * 1, 2,
      2 + 3 * 1 - 0.5 * 1,    2 + 3 * 2 - 0.5 * 4,    2 + 3 * 3 - 0.5 * 9,
      2 + 3 * 4 - 0.5 * 16};

  memcpy(X.data, x_vals, sizeof(x_vals));
  memcpy(Y.data, y_vals, sizeof(y_vals));

  // Construction de la matrice de design polynomiale (avec normalisation)
  f32 mean_x, std_x;
  Table X_poly = build_poly2_design(&X, &mean_x, &std_x);

  printf("Donnees normalisees: mean=%.3f, std=%.3f\n", mean_x, std_x);
  printf("Matrice de design (3 premieres lignes):\n");
  for (uint i = 0; i < 3 && i < n_samples; i++) {
    printf("  [1, %.3f, %.3f] -> y=%.3f\n", table_get(&X_poly, i, 1),
           table_get(&X_poly, i, 2), table_get(&Y, i, 0));
  }

  // Entraînement par descente de gradient
  PolyRegression model = {0.0f, 0.0f, 0.0f};
  poly_gradient_descent(&model, &X_poly, &Y, 0.1f, 5000);

  // Coefficients dans l'espace normalise
  printf("\nCoefficients (features normalisees):\n");
  printf("  w0 (biais)        = %f\n", model.w0);
  printf("  w1 (coeff x')     = %f\n", model.w1);
  printf("  w2 (coeff x'^2)   = %f\n", model.w2);

  // Reconversion vers les coefficients du polynôme original y = c₀ + c₁x + c₂x²
  // En developpant : ŷ = w₀ + w₁·(x-μ)/σ + w₂·((x-μ)/σ)²
  // On obtient :
  //   c₂ = w₂ / σ²
  //   c₁ = w₁/σ - 2·w₂·μ/σ²
  //   c₀ = w₀ - w₁·μ/σ + w₂·μ²/σ²
  f32 c2 = model.w2 / (std_x * std_x);
  f32 c1 = model.w1 / std_x - 2.0f * model.w2 * mean_x / (std_x * std_x);
  f32 c0 = model.w0 - model.w1 * mean_x / std_x +
           model.w2 * mean_x * mean_x / (std_x * std_x);

  printf("\nPolynome original: y = %.4f + %.4f x + %.4f x^2\n", c0, c1, c2);

  // Prediction pour un nouveau x
  f32 new_x = 2.5f;
  f32 pred = poly_predict(&model, new_x, mean_x, std_x);
  printf("Prediction pour x=%.2f: %.4f\n", new_x, pred);

  // Nettoyage
  free_table(&X);
  free_table(&Y);
  free_table(&X_poly);

  return 0;
}
