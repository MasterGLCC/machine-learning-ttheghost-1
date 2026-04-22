#include <common/math.h>
#include <common/csv.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// Definition d'un voisin pour KNN
typedef struct
{
  f32 distance;
  f32 label;
} Neighbor;

// Comparateur pour qsort : trie par distance croissante
int compare_neighbors(const void *a, const void *b)
{
  Neighbor *n1 = (Neighbor *)a;
  Neighbor *n2 = (Neighbor *)b;
  if (n1->distance > n2->distance)
    return 1;
  if (n1->distance < n2->distance)
    return -1;
  return 0;
}

// Calcul de la distance euclidienne entre un point de train_set et le query_point
// train_set : n×m (n exemples, m features)
// query_point : 1×m (point a classer)
f32 euclidean_distance(const Table *train_set, uint row_idx, const Table *query_point)
{
  f32 sum = 0.0f;
  f32 *restrict train_row = &train_set->data[row_idx * train_set->cols];
  f32 *restrict query_row = query_point->data;
  for (uint j = 0; j < train_set->cols; j++)
  {
    f32 diff = train_row[j] - query_row[j];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

// Prediction KNN : retourne la classe majoritaire parmi les k plus proches voisins
f32 knn_predict(const Table *X_train, const Table *y_train, const Table *query, uint k)
{
  uint n_samples = X_train->rows;
  Neighbor *neighbors = malloc(n_samples * sizeof(Neighbor));

  for (uint i = 0; i < n_samples; i++)
  {
    neighbors[i].distance = euclidean_distance(X_train, i, query);
    neighbors[i].label = table_get(y_train, i, 0);
  }

  // Tri des voisins par distance
  qsort(neighbors, n_samples, sizeof(Neighbor), compare_neighbors);

  f32 best_label = -1.0f;
  int max_count = -1;

  for (uint i = 0; i < k; i++)
  {
    int count = 0;
    for (uint j = 0; j < k; j++)
    {
      if (neighbors[j].label == neighbors[i].label)
      {
        count++;
      }
    }
    if (count > max_count)
    {
      max_count = count;
      best_label = neighbors[i].label;
    }
  }

  free(neighbors);
  return best_label;
}

// Calcule l'accuracy du modèle KNN sur un ensemble de validation
f32 calculate_accuracy(const Table *X_train, const Table *y_train,
                       const Table *X_val, const Table *y_val, uint k)
{
  uint correct = 0;
  Table query = init_table(1, X_train->cols); // 1xm pour stocker un point de validation à la fois

  for (uint i = 0; i < X_val->rows; i++)
  {
    // Remplir le query point avec les features de l'exemple de validation i
    for (uint j = 0; j < X_val->cols; j++)
    {
      table_set(&query, 0, j, table_get(X_val, i, j));
    }

    f32 pred = knn_predict(X_train, y_train, &query, k);
    f32 actual = table_get(y_val, i, 0);

    if (pred == actual)
    {
      correct++;
    }
  }

  free_table(&query);
  return (f32)correct / X_val->rows;
}

// Implementation de la validation croisee K-fold pour KNN
f32 cross_validate_knn(Table *X, Table *y, uint k_folds, uint knn_k)
{
  // Shuffle les donnees avant de les diviser en folds
  table_shuffle_together(X, y);

  uint samples_per_fold = X->rows / k_folds;
  float total_accuracy = 0.0f;

  for (uint i = 0; i < k_folds; i++)
  {
    uint start_val = i * samples_per_fold;

    // Pour le dernier fold, on prend tous les echantillons restants pour eviter les problèmes de division
    uint end_val = (i == k_folds - 1) ? X->rows : (i + 1) * samples_per_fold;

    // Create Validation Set (fold i)
    Table X_val = table_extract_rows(X, start_val, end_val);
    Table y_val = table_extract_rows(y, start_val, end_val);

    // Create Training Set (tous les autres folds combines)
    Table X_train = table_combine_except(X, start_val, end_val);
    Table y_train = table_combine_except(y, start_val, end_val);

    // Calcul de l'accuracy pour ce fold
    float fold_acc = calculate_accuracy(&X_train, &y_train, &X_val, &y_val, knn_k);
    total_accuracy += fold_acc;

    // Clean up
    free_table(&X_val);
    free_table(&y_val);
    free_table(&X_train);
    free_table(&y_train);
  }

  printf("Pour K=%d, Precision moyenne : %.2f%%\n", knn_k, (total_accuracy / k_folds) * 100);
  return total_accuracy / k_folds;
}

// Trouver le K optimal en testant differentes valeurs de K et en utilisant la validation croisee
uint find_optimal_k(Table *X, Table *y, uint k_folds, uint max_k)
{
  printf("--- Recherche du K optimal pour KNN ---\n");
  uint best_k = 1;
  float best_accuracy = 0.0f;

  for (uint k = 1; k <= max_k; k++)
  {
    float avg_acc = cross_validate_knn(X, y, k_folds, k);
    if (avg_acc > best_accuracy)
    {
      best_accuracy = avg_acc;
      best_k = k;
    }
  }

  printf("K optimal trouve : %d avec une precision de %.2f%%\n", best_k, best_accuracy * 100);
  return best_k;
}

/*
Resultats attendus:

Charge './datasets/iris.csv': 150 lignes x 5 colonnes
--- Recherche du K optimal pour KNN ---
Pour K=1, Precision moyenne : 94.67%
Pour K=2, Precision moyenne : 94.67%
Pour K=3, Precision moyenne : 94.67%
Pour K=4, Precision moyenne : 95.33%
Pour K=5, Precision moyenne : 96.00%
Pour K=6, Precision moyenne : 95.33%
Pour K=7, Precision moyenne : 96.67%
Pour K=8, Precision moyenne : 94.67%
Pour K=9, Precision moyenne : 94.00%
Pour K=10, Precision moyenne : 96.00%
K optimal trouve : 7 avec une precision de 96.67%
La classe predite pour l'exemple [5.1, 3.5, 1.4, 0.2] est : setosa
*/

int main()
{
  srand(42); // Seed pour la reproductibilite

  // Charger le jeu de donnees Iris depuis un fichier CSV
  // setosa: 0, versicolor: 1, virginica: 2
  Table dataset = table_load_csv("./datasets/iris.csv", 1);

  if (dataset.data == NULL)
  {
    printf("Erreur lors du chargement du dataset iris.\n");
    return 1;
  }
  
  Table X_unnormalized = table_extract_columns(&dataset, 0, 4);
  Table y = table_extract_column(&dataset, 4);

  Table mean = table_mean_axis0(&X_unnormalized);
  Table stddev = table_stddev_axis0(&X_unnormalized, &mean);

  Table X = table_copy(&X_unnormalized);
  table_normlize_zscore_axis0(&X, &mean, &stddev);

  // Trouver le K optimal
  uint k = find_optimal_k(&X, &y, 5, 10);

  // Predire la classe d'un nouvel exemple (ex: [5.1, 3.5, 1.4, 0.2] qui correspond à setosa)
  Table query = init_table(1, X.cols);
  f32 data[] = {5.1f, 3.5f, 1.4f, 0.2f};
  memcpy(query.data, data, sizeof(data));
  table_normlize_zscore_axis0(&query, &mean, &stddev);

  f32 predicted_label = knn_predict(&X, &y, &query, k);
  char* predicted_label_str = (predicted_label == 0.0f) ? "setosa" : (predicted_label == 1.0f) ? "versicolor" : "virginica";
  printf("La classe predite pour l'exemple [5.1, 3.5, 1.4, 0.2] est : %s\n", predicted_label_str);

  // Clean up
  free_table(&dataset);
  free_table(&X_unnormalized);
  free_table(&X);
  free_table(&y);
  free_table(&mean);
  free_table(&stddev);
  return 0;
}