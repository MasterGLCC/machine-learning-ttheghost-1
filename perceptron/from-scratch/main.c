#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <common/math.h>
#include <common/matrix.h>
#include <common/csv.h>

typedef struct
{
  // Poids du modele : W de dimension (n_features, 1)
  Table weights;
  f32 bias;
} Perceptron;

// Fonction d'activation sigmoide : σ(z) = 1 / (1 + e⁻ᶻ)
f32 sigmoid(f32 x)
{
  return 1.0f / (1.0f + expf(-x));
}

// Prediction du modele pour une entree x : ŷ = σ(∑ (w_j * x_j) + b)
f32 model_predict(Perceptron *model, Table *input)
{
  f32 sum = model->bias;
  for (uint i = 0; i < input->cols; i++)
  {
    f32 w = table_get(&model->weights, i, 0);
    f32 x = table_get(input, 0, i);
    sum += w * x;
  }
  return sigmoid(sum);
}

// Predictions pour un lot d'entrees X : Y_pred = σ(X·W + b)
// Retourne un tableau de dimension (n_echantillons, 1)
Table batch_model_predict(Perceptron *model, Table *input)
{
  Table output = init_table(input->rows, 1);
  assert(input->cols == model->weights.rows);
  for (uint i = 0; i < input->rows; i++)
  {
    Table input_row = table_extract_row(input, i);
    table_set(&output, i, 0, model_predict(model, &input_row));
    free_table(&input_row);
  }
  return output;
}

// Entrainement par descente de gradient avec perte d'entropie croisee binaire
// Perte : L = -1/N * ∑ [ y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i) ]
// Gradients : ∂L/∂W = 1/N * Xᵀ · (ŷ - y) et ∂L/∂b = 1/N * ∑ (ŷ_i - y_i)
void train_model(Perceptron *model, Table *X, Table *y, f32 lr, uint epochs)
{
  for (size_t epoch = 0; epoch < epochs; epoch++)
  {
    // Etape de propagation avant : ŷ = σ(X·W + b)
    Table pred = batch_model_predict(model, X);

    // Calcul de l'erreur : e = ŷ - y
    Table error = table_sub(&pred, y);

    // Calcul du gradient moyen des poids et du biais par rapport a la perte L
    Table XT = matrix_transpose(X);
    Table _dW = matrix_multiply(&XT, &error);
    Table dW = table_div_scalar(&_dW, (f32)X->rows);
    f32 db = table_sum(&error) / (f32)X->rows;

    // Mise a jour des parametres : W = W - α * ∂L/∂W et b = b - α * ∂L/∂b
    for (uint i = 0; i < model->weights.rows; i++)
    {
      f32 w = table_get(&model->weights, i, 0);
      f32 dw = table_get(&dW, i, 0);
      table_set(&model->weights, i, 0, w - lr * dw);
    }
    model->bias -= lr * db;

    free_table(&pred);
    free_table(&error);
    free_table(&XT);
    free_table(&_dW);
    free_table(&dW);
  }
}

int main()
{
  printf("-- Perceptron --\n");
  Table data = table_load_csv("datasets/perceptron_data.csv", 0);
  Table X = table_extract_columns(&data, 0, data.cols - 1);
  Table y = table_extract_column(&data, data.cols - 1);

  printf("Dataset charge : %u echantillons, %u features.\n", X.rows, X.cols);

  Perceptron model = {
    .weights = init_table_with(X.cols, 1, 0), // Initialisation des poids W a zero
    .bias = 0.0f
  };

  train_model(&model, &X, &y, 0.1f, 1000);

  // Affichage des predictions finales
  for (uint i = 0; i < X.rows; i++) {
    Table input_row = table_extract_row(&X, i);
    f32 pred = model_predict(&model, &input_row);
    printf("Echantillon %2u : Vraie Classe = %.0f, Prediction = %.4f\n", i,
           table_get(&y, i, 0), pred);
    free_table(&input_row);
  }

  free_table(&X);
  free_table(&y);
  free_table(&model.weights);
  free_table(&data);
  return 0;
}

// Affichage attendu :

// -- Perceptron --
// Charge 'datasets/perceptron_data.csv': 5 lignes x 3 colonnes
// Dataset charge : 5 echantillons, 2 features.
// Echantillon  0 : Vraie Classe = 0, Prediction = 0.1145
// Echantillon  1 : Vraie Classe = 0, Prediction = 0.1145
// Echantillon  2 : Vraie Classe = 1, Prediction = 0.9698
// Echantillon  3 : Vraie Classe = 1, Prediction = 0.9698
// Echantillon  4 : Vraie Classe = 1, Prediction = 0.9078