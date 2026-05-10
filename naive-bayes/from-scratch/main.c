/**
 * Mohammed IFKIRNE
 */

/**
 * Implementation de Naive Bayes Gaussien (Gaussian Naive Bayes)
 *
 * Note importante: Naive Bayes est un algorithme d'apprentissage SUPERVISE
 * (il a besoin d'etiquettes/classes cibles 'y'), contrairement a DBSCAN qui est
 * non-supervise.
 *
 * Cet algorithme se base sur le Theoreme de Bayes avec une hypothese "naive"
 * d'independance conditionnelle entre les caracteristiques (features) etant
 * donnee la classe.
 *
 * Concepts mathematiques cles:
 * 1. Theoreme de Bayes : P(C_k | X) = [P(X | C_k) * P(C_k)] / P(X)
 *    Puisque P(X) est constant pour toutes les classes, on cherche a maximiser:
 *    P(C_k | X) = P(C_k) * Produit_i( P(x_i | C_k) )
 *
 * 2. Hypothese Gaussienne (Loi Normale) : Pour des variables continues, on
 *    suppose que les donnees de chaque classe suivent une distribution normale
 * : P(x_i | C_k) = (1 / sqrt(2 * pi * sigma^2)) * exp( - (x_i - mu)^2 / (2 *
 * sigma^2) )
 *
 * En pratique, pour eviter le sous-depassement numerique (underflow) quand on
 * multiplie de petites probabilites, on utilise le Logarithme des probabilites
 * (Log-Probabilites). log(P(C_k | X)) = log(P(C_k)) + Somme_i( log( P(x_i |
 * C_k) ) )
 */

#include "common/csv.h"
#include "common/math.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265358979323846f

// Structure pour stocker les parametres appris par classe
typedef struct {
  f32 prior_prob; // P(C_k)
  f32 log_prior;  // log(P(C_k))
  f32 *means;     // mu_i (Moyennes des features pour cette classe)
  f32 *variances; // sigma^2_i (Variances des features pour cette classe)
} GaussianNBClass;

typedef struct {
  GaussianNBClass *classes;
  f32 *unique_labels;
  uint num_classes;
  uint num_features;
} GaussianNB;

// Trouve toutes les classes uniques (labels)
f32 *get_unique_labels_nb(const Table *Y, uint *num_unique) {
  f32 *uniques = malloc(Y->rows * sizeof(f32));
  uint count = 0;
  for (uint i = 0; i < Y->rows; i++) {
    f32 val = table_get(Y, i, 0);
    bool found = false;
    for (uint j = 0; j < count; j++) {
      if (uniques[j] == val) {
        found = true;
        break;
      }
    }
    if (!found)
      uniques[count++] = val;
  }
  *num_unique = count;
  return uniques;
}

// Fonction pour entrainer le modele
GaussianNB fit_naive_bayes(const Table *X, const Table *Y) {
  GaussianNB model;
  model.num_features = X->cols;
  model.unique_labels = get_unique_labels_nb(Y, &model.num_classes);
  model.classes = malloc(model.num_classes * sizeof(GaussianNBClass));

  // Pour chaque classe k
  for (uint k = 0; k < model.num_classes; k++) {
    f32 current_class = model.unique_labels[k];

    // 1. Calcul de la probabilite a priori P(C_k)
    uint class_count = 0;
    for (uint i = 0; i < Y->rows; i++) {
      if (table_get(Y, i, 0) == current_class)
        class_count++;
    }
    model.classes[k].prior_prob = (f32)class_count / Y->rows;
    model.classes[k].log_prior = logf(model.classes[k].prior_prob);

    // 2. Calcul de la moyenne (mu) et de la variance (sigma^2) pour chaque
    // feature
    model.classes[k].means = calloc(model.num_features, sizeof(f32));
    model.classes[k].variances = calloc(model.num_features, sizeof(f32));

    // Moyenne
    for (uint j = 0; j < model.num_features; j++) {
      f32 sum = 0.0f;
      for (uint i = 0; i < X->rows; i++) {
        if (table_get(Y, i, 0) == current_class) {
          sum += table_get(X, i, j);
        }
      }
      model.classes[k].means[j] = sum / class_count;
    }

    // Variance
    for (uint j = 0; j < model.num_features; j++) {
      f32 sum_sq_diff = 0.0f;
      for (uint i = 0; i < X->rows; i++) {
        if (table_get(Y, i, 0) == current_class) {
          f32 diff = table_get(X, i, j) - model.classes[k].means[j];
          sum_sq_diff += diff * diff;
        }
      }
      // On rajoute un petit epsilon (1e-6) pour eviter une variance de 0
      // (division par zero)
      model.classes[k].variances[j] = (sum_sq_diff / class_count) + 1e-6f;
    }
  }

  return model;
}

// Calcule la log-probabilite de densite Gaussienne : log( P(x_i | C_k) )
f32 log_gaussian_pdf(f32 x, f32 mean, f32 var) {
  f32 log_coeff = -0.5f * logf(2.0f * PI * var);
  f32 exponent = -((x - mean) * (x - mean)) / (2.0f * var);
  return log_coeff + exponent;
}

// Predire la classe d'une seule ligne
f32 predict_naive_bayes(const GaussianNB *model, const Table *X, uint row) {
  f32 best_log_prob = -INFINITY;
  f32 best_class = -1.0f;

  for (uint k = 0; k < model->num_classes; k++) {
    // Commence avec log(P(C_k))
    f32 log_prob = model->classes[k].log_prior;

    // Ajoute log(P(x_i | C_k)) pour chaque feature
    for (uint j = 0; j < model->num_features; j++) {
      f32 x_val = table_get(X, row, j);
      f32 mean = model->classes[k].means[j];
      f32 var = model->classes[k].variances[j];

      log_prob += log_gaussian_pdf(x_val, mean, var);
    }

    // Trouver le Maximum a posteriori (MAP)
    if (log_prob > best_log_prob) {
      best_log_prob = log_prob;
      best_class = model->unique_labels[k];
    }
  }

  return best_class;
}

// Libere la memoire du modele
void free_naive_bayes(GaussianNB *model) {
  for (uint k = 0; k < model->num_classes; k++) {
    free(model->classes[k].means);
    free(model->classes[k].variances);
  }
  free(model->classes);
  free(model->unique_labels);
}

int main() {
  printf("--- Naive Bayes Gaussien ---\n");

  // Chargement du dataset (Continuous features, Discrete labels)
  Table data = table_load_csv("datasets/nb_data.csv", 1);
  Table X = table_extract_columns(&data, 0, data.cols - 1);
  Table Y = table_extract_column(&data, data.cols - 1);

  printf("Dataset charge : %u echantillons, %u features.\n", X.rows, X.cols);

  // 1. Entrainement
  GaussianNB model = fit_naive_bayes(&X, &Y);
  printf("Modele entraine. Classes trouvees : %u\n\n", model.num_classes);

  // 2. Prediction et Evaluation
  for (uint i = 0; i < X.rows; i++) {
    f32 y_pred = predict_naive_bayes(&model, &X, i);
    f32 y_true = table_get(&Y, i, 0);

    // Affichage des premieres predictions pour illustrer
    if (i < 5 || i >= X.rows - 5) {
      printf("Echantillon %2u : Vraie Classe = %.0f, Prediction = %.0f\n", i,
             y_true, y_pred);
    }
    if (i == 5 && X.rows > 10) {
      printf("...\n");
    }
  }

  // 3. Nettoyage
  free_naive_bayes(&model);
  free_table(&X);
  free_table(&Y);
  free_table(&data);

  return 0;
}

// Affichage du code:

// Charge 'datasets/nb_data.csv': 10 lignes x 3 colonnes
// Dataset charge : 10 echantillons, 2 features.
// Modele entraine. Classes trouvees : 2

// Echantillon  0 : Vraie Classe = 0, Prediction = 0
// Echantillon  1 : Vraie Classe = 0, Prediction = 0
// Echantillon  2 : Vraie Classe = 0, Prediction = 0
// Echantillon  3 : Vraie Classe = 0, Prediction = 0
// Echantillon  4 : Vraie Classe = 0, Prediction = 0
// Echantillon  5 : Vraie Classe = 1, Prediction = 1
// Echantillon  6 : Vraie Classe = 1, Prediction = 1
// Echantillon  7 : Vraie Classe = 1, Prediction = 1
// Echantillon  8 : Vraie Classe = 1, Prediction = 1
// Echantillon  9 : Vraie Classe = 1, Prediction = 1