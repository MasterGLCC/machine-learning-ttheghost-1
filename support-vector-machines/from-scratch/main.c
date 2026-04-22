#include <common/math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Structure pour stocker le modele SVM
typedef struct
{
  Table w; // Poids (1 x colonnes)
  f32 b;   // Biais
} LinearSVM;

// Entrainement du SVM lineaire
LinearSVM svm_train(const Table *X, const Table *y, f32 C, f32 learning_rate, uint epochs)
{
  LinearSVM model;
  model.w = init_table(1, X->cols);
  model.b = 0.0f;

  // Initialisation des poids à 0
  for (uint j = 0; j < X->cols; j++)
  {
    table_set(&model.w, 0, j, 0.0f);
  }

  for (uint epoch = 0; epoch < epochs; epoch++)
  {
    for (uint i = 0; i < X->rows; i++)
    {
      // 1. Calculer le produit scalaire : w * x_i
      f32 dot_product = 0.0f;
      for (uint j = 0; j < X->cols; j++)
      {
        dot_product += table_get(&model.w, 0, j) * table_get(X, i, j);
      }

      // 2. Le SVM exige des labels de classe {-1, 1}
      // Si votre dataset (ex: Iris) utilise {0, 1}, on convertit temporairement :
      f32 y_val = table_get(y, i, 0);
      f32 y_i = (y_val == 0.0f) ? -1.0f : 1.0f;

      // 3. Verification de la condition de Marge (Hinge Loss)
      if (y_i * (dot_product + model.b) >= 1.0f)
      {
        // Point correctement classe en dehors de la marge :
        // On applique uniquement la penalite de regularisation pour maximiser la marge
        for (uint j = 0; j < X->cols; j++)
        {
          f32 current_w = table_get(&model.w, 0, j);
          table_set(&model.w, 0, j, current_w - learning_rate * current_w);
        }
      }
      else
      {
        // Point dans la marge ou mal classe (Violation de la Soft Margin) :
        // On met à jour w et b en tenant compte de l'erreur et du parametre C
        for (uint j = 0; j < X->cols; j++)
        {
          f32 current_w = table_get(&model.w, 0, j);
          f32 x_ij = table_get(X, i, j);
          f32 gradient = current_w - C * y_i * x_ij;
          table_set(&model.w, 0, j, current_w - learning_rate * gradient);
        }
        model.b = model.b + learning_rate * C * y_i;
      }
    }
  }
  return model;
}

// Prediction avec le modele SVM entraine
f32 svm_predict(const LinearSVM *model, const Table *query)
{
  f32 dot_product = 0.0f;
  for (uint j = 0; j < model->w.cols; j++)
  {
    dot_product += table_get(&model->w, 0, j) * table_get(query, 0, j);
  }

  // Le côte de l'hyperplan determine la classe
  f32 prediction = dot_product + model->b;

  // On retourne 1.0f ou 0.0f pour correspondre à notre format de donnees d'origine
  return (prediction >= 0.0f) ? 1.0f : 0.0f;
}

int main()
{
  srand(42);

  // Creation de donnees fictives (Separables avec un peu de bruit)
  uint n_samples = 200;
  Table X = init_table(n_samples, 2);
  Table y = init_table(n_samples, 1);

  printf("Generation de donnees d'entrainement...\n");
  for (uint i = 0; i < n_samples; i++)
  {
    if (i < n_samples / 2)
    {
      // Classe 0 (en bas à gauche)
      table_set(&X, i, 0, (f32)(rand() % 40) / 10.0f); // x entre 0 et 4
      table_set(&X, i, 1, (f32)(rand() % 40) / 10.0f); // y entre 0 et 4
      table_set(&y, i, 0, 0.0f);
    }
    else
    {
      // Classe 1 (en haut à droite)
      table_set(&X, i, 0, 3.0f + (f32)(rand() % 40) / 10.0f); // x entre 3 et 7
      table_set(&X, i, 1, 3.0f + (f32)(rand() % 40) / 10.0f); // y entre 3 et 7
      table_set(&y, i, 0, 1.0f);
    }
  }

  // Parametres de l'entrainement
  f32 C = 1.0f;              // Parametre de Marge Douce
  f32 learning_rate = 0.01f; // Taux d'apprentissage
  uint epochs = 1000;        // Nombre de passages sur les donnees

  printf("Entrainement du modele SVM (C=%.2f, epochs=%d)...\n", C, epochs);
  LinearSVM model = svm_train(&X, &y, C, learning_rate, epochs);

  printf("Entrainement termine.\n");
  printf("Poids trouves : W1 = %.3f, W2 = %.3f\n", table_get(&model.w, 0, 0), table_get(&model.w, 0, 1));
  printf("Biais trouve  : b = %.3f\n\n", model.b);

  // evaluation rapide sur les donnees d'entrainement
  uint correct = 0;
  Table query = init_table(1, 2);

  for (uint i = 0; i < n_samples; i++)
  {
    table_set(&query, 0, 0, table_get(&X, i, 0));
    table_set(&query, 0, 1, table_get(&X, i, 1));

    f32 pred = svm_predict(&model, &query);
    f32 actual = table_get(&y, i, 0);

    if (pred == actual)
      correct++;
  }

  printf("Precision sur l'ensemble d'entrainement : %.2f%%\n", ((float)correct / n_samples) * 100);

  // Liberation de la memoire
  free_table(&X);
  free_table(&y);
  free_table(&model.w);
  free_table(&query);

  return 0;
}