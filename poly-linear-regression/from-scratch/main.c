/**
 * Mohammed IFKIRNE
 */

 #include <common/csv.h>
#include <common/math.h>
#include <math.h>
#include <stdio.h>

// Regression polynomiale de degre 2 : ŷ = w₀ + w₁·x + w₂·x²
typedef struct {
  f32 w0, w1, w2;
} PolyRegression;

// Construction de la matrice de design polynomiale (degre 2)
// X doit etre deja normalise (z-score)
// Colonnes resultantes : [1, x', x'²]
Table build_poly2_design(const Table *X) {
  Table X_poly = init_table(X->rows, 3);
  // Remplissage : colonne 0 = 1 (biais), colonne 1 = x', colonne 2 = x'²
  for (uint i = 0; i < X->rows; i++) {
    f32 x = table_get(X, i, 0);
    table_set(&X_poly, i, 0, 1.0f);
    table_set(&X_poly, i, 1, x);
    table_set(&X_poly, i, 2, x * x);
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
  // Chargement du dataset taille/poids
  Table c = table_load_csv("datasets/SOCR-HeightWeight.csv", 1);

  // Normalisation z-score pour stabiliser la descente de gradient
  Table mean = table_mean_axis0(&c);
  Table stddev = table_stddev_axis0(&c, &mean);
  table_normlize_zscore_axis0(&c, &mean, &stddev);

  // X = taille (colonne 1), Y = poids (colonne 2) -- deja normalises
  Table X = table_extract_column(&c, 1);
  Table Y = table_extract_column(&c, 2);

  // Construction de la matrice de design polynomiale [1, x', x'²]
  Table X_poly = build_poly2_design(&X);

  // Entrainement par descente de gradient
  PolyRegression model = {0.0f, 0.0f, 0.0f};
  poly_gradient_descent(&model, &X_poly, &Y, 0.1f, 5000);

  // Coefficients dans l'espace normalise
  printf("\nCoefficients (features normalisees):\n");
  printf("  w0 (biais)        = %f\n", model.w0);
  printf("  w1 (coeff x')     = %f\n", model.w1);
  printf("  w2 (coeff x'^2)   = %f\n", model.w2);

  f32 mean_x = table_get(&mean, 0, 1);
  f32 mean_y = table_get(&mean, 0, 2);
  f32 std_x = table_get(&stddev, 0, 1);
  f32 std_y = table_get(&stddev, 0, 2);

  // Reconversion vers l'echelle originale y = c₀ + c₁x + c₂x²
  // Le modele predit y' (normalise), donc y_orig = y'·σ_y + μ_y
  // Et x' = (x - μ_x) / σ_x
  // En combinant : y_orig = σ_y·(w₀ + w₁·x' + w₂·x'²) + μ_y
  // En developpant sur x original :
  //   c₂ = σ_y · w₂ / σ_x²
  //   c₁ = σ_y · (w₁/σ_x - 2·w₂·μ_x/σ_x²)
  //   c₀ = σ_y · (w₀ - w₁·μ_x/σ_x + w₂·μ_x²/σ_x²) + μ_y
  f32 c2 = std_y * model.w2 / (std_x * std_x);
  f32 c1 =
      std_y * (model.w1 / std_x - 2.0f * model.w2 * mean_x / (std_x * std_x));
  f32 c0 = std_y * (model.w0 - model.w1 * mean_x / std_x +
                    model.w2 * mean_x * mean_x / (std_x * std_x)) +
           mean_y;

  printf("\nPolynome original: y = %.4f + %.4f x + %.4f x^2\n", c0, c1, c2);

  // Prediction pour un nouveau x (dans l'echelle originale)
  f32 new_x = 68.0f; // taille en pouces
  f32 x_norm = (new_x - mean_x) / std_x;
  f32 pred_norm = model.w0 + model.w1 * x_norm + model.w2 * (x_norm * x_norm);
  f32 pred_orig = pred_norm * std_y + mean_y;
  printf("Prediction pour taille=%.1f pouces: poids=%.2f lbs\n", new_x, pred_orig);

  // Nettoyage
  free_table(&X);
  free_table(&Y);
  free_table(&X_poly);
  free_table(&mean);
  free_table(&stddev);
  free_table(&c);

  return 0;
}
