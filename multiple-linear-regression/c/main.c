#include <common/csv.h>
#include <common/math.h>
#include <common/matrix.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Régression linéaire multiple : ŷ = Xw
// On utilise l'équation normale pour trouver w
typedef struct {
  Table weights;
} LinearRegression;

// Ajoute une colonne de 1 au début de X pour le terme d'ordonnée à l'origine
// (biais) X_features (n×m) → X_design (n×(m+1)) avec la premiere colonne = 1
Table add_intercept_column(const Table *X_features) {
  Table X = init_table(X_features->rows, X_features->cols + 1);
  for (uint i = 0; i < X_features->rows; i++) {
    table_set(&X, i, 0, 1.0f); // colonne de biais
    for (uint j = 0; j < X_features->cols; j++) {
      table_set(&X, i, j + 1, table_get(X_features, i, j));
    }
  }
  return X;
}

// Entraînement par l'équation normale (solution analytique)
// w = (XᵀX)⁻¹ · Xᵀy
void linear_regression_fit(LinearRegression *model, const Table *X_features,
                           const Table *Y) {
  if (X_features->rows != Y->rows || Y->cols != 1) {
    fprintf(stderr, "fit: dimensions incompatibles\n");
    exit(EXIT_FAILURE);
  }

  // Construction de la matrice de design avec colonne d'intercept
  Table X_design = add_intercept_column(X_features);

  // Calcul de Xᵀ
  Table Xt = matrix_transpose(&X_design);

  // Calcul de XᵀX
  Table XtX = matrix_multiply(&Xt, &X_design);

  // Calcul de Xᵀy
  Table XtY = matrix_multiply(&Xt, Y);

  // Inversion de (XᵀX)
  Table XtX_inv = matrix_inverse(&XtX);

  // Résultat : w = (XᵀX)⁻¹ · Xᵀy   → vecteur (m+1) × 1
  Table W = matrix_multiply(&XtX_inv, &XtY);

  model->weights = W;

  free_table(&X_design);
  free_table(&Xt);
  free_table(&XtX);
  free_table(&XtY);
  free_table(&XtX_inv);
}

// Prédiction : ŷ = X_design · w
Table linear_regression_predict(const LinearRegression *model,
                                const Table *X_features) {
  Table X_design = add_intercept_column(X_features);
  Table Y_pred = matrix_multiply(&X_design, &model->weights);
  free_table(&X_design);
  return Y_pred;
}

void linear_regression_free(LinearRegression *model) {
  if (model->weights.data)
    free_table(&model->weights);
  model->weights.data = NULL;
  model->weights.rows = 0;
  model->weights.cols = 0;
}

int main() {
  // Exemple : prédire le prix d'une maison à partir de la surface et nb de
  // chambres
  Table X_features = init_table(5, 2);
  f32 X_data[] = {650, 1, // surface (sqft), chambres
                  780, 2, 920, 3, 1100, 3, 1350, 4};
  memcpy(X_features.data, X_data, sizeof(X_data));

  Table Y = init_table(5, 1);
  f32 Y_data[] = {130000, 160000, 195000, 230000, 280000};
  memcpy(Y.data, Y_data, sizeof(Y_data));

  // Entraînement via équation normale
  LinearRegression model = {0};
  linear_regression_fit(&model, &X_features, &Y);

  // Affichage des poids trouvés
  printf("Intercept (biais):    %f\n", table_get(&model.weights, 0, 0));
  printf("Coefficient surface:  %f\n", table_get(&model.weights, 1, 0));
  printf("Coefficient chambres: %f\n", table_get(&model.weights, 2, 0));

  // Prédiction pour une nouvelle maison : 1000 sqft, 3 chambres
  Table X_new = init_table(1, 2);
  table_set(&X_new, 0, 0, 1000);
  table_set(&X_new, 0, 1, 3);
  Table Y_new = linear_regression_predict(&model, &X_new);
  printf("Prix prédit: %f\n", table_get(&Y_new, 0, 0));

  // Nettoyage mémoire
  free_table(&X_features);
  free_table(&Y);
  free_table(&X_new);
  free_table(&Y_new);
  linear_regression_free(&model);

  return 0;
}
