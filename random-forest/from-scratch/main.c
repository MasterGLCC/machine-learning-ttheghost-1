/**
 * Mohammed IFKIRNE
 */

/**
 * Implementation de la Foret Aleatoire (Random Forest - Bootstrap Aggregating)
 *
 * Le principe du Bootstrap Aggregating (Bagging) est d'entrainer de multiples
 * arbres de decision sur des sous-ensembles des donnees tires aleatoirement
 * avec remise (bootstrap).
 *
 * Concepts mathematiques cles:
 * 1. Bootstrap: Tirage avec remise de N echantillons a partir du jeu de donnees
 *    original de taille N. Environ 63.2% des donnees uniques sont selectionnees
 *    dans chaque echantillon, le reste etant des doublons.
 *
 * 2. Aggregation (Vote Majoritaire): Pour une nouvelle donnee, chaque arbre
 *    donne sa prediction. La classe finale est celle qui a recu le plus de
 *    votes parmi les arbres de la foret.
 *    y_pred = argmax_c ( Somme( I(h_k(x) == c) ) )
 *    ou h_k est le k-ieme arbre et I est la fonction indicatrice (1 si vrai, 0 sinon).
 */

#include "common/csv.h"
#include "common/math.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Structure d'un noeud de l'arbre de decision
typedef struct TreeNode {
  bool is_leaf;      // Vrai si c'est une feuille (decision finale)
  f32 label;         // Classe predite (valide uniquement si is_leaf = true)
  int feature_idx;   // Index de l'attribut utilise pour la separation
  f32 *split_values; // Valeurs possibles de l'attribut (branches)
  struct TreeNode **children; // Sous-arbres correspondant a chaque valeur
  uint num_children;          // Nombre de branches (enfants)
} TreeNode;

// Structure de la Foret Aleatoire
typedef struct {
  TreeNode **trees; // Tableau de pointeurs vers les racines des arbres
  uint num_trees;   // Nombre d'arbres dans la foret
} RandomForest;

// Fonction utilitaire pour filtrer la table selon une valeur specifique d'une
// colonne
Table table_filter(const Table *X, const Table *Y, uint col, f32 val,
                   Table *out_Y) {
  uint count = 0;
  for (uint i = 0; i < X->rows; i++) {
    if (table_get(X, i, col) == val)
      count++;
  }

  Table res_X = init_table(count, X->cols);
  *out_Y = init_table(count, Y->cols);

  uint idx = 0;
  for (uint i = 0; i < X->rows; i++) {
    if (table_get(X, i, col) == val) {
      for (uint j = 0; j < X->cols; j++) {
        table_set(&res_X, idx, j, table_get(X, i, j));
      }
      for (uint j = 0; j < Y->cols; j++) {
        table_set(out_Y, idx, j, table_get(Y, i, j));
      }
      idx++;
    }
  }
  return res_X;
}

// Trouve toutes les valeurs uniques presentes dans une colonne (attribut)
f32 *get_unique_values(const Table *t, uint col, uint *num_unique) {
  f32 *uniques = malloc(t->rows * sizeof(f32));
  uint count = 0;
  for (uint i = 0; i < t->rows; i++) {
    f32 val = table_get(t, i, col);
    bool found = false;
    for (uint j = 0; j < count; j++) {
      if (uniques[j] == val) {
        found = true;
        break;
      }
    }
    if (!found) {
      uniques[count++] = val;
    }
  }
  *num_unique = count;
  return uniques;
}

// Calcul de l'Entropie : H(S) = - Somme(p_i * log2(p_i))
f32 calc_entropy(const Table *Y) {
  if (Y->rows == 0)
    return 0.0f;
  uint num_unique;
  f32 *uniques = get_unique_values(Y, 0, &num_unique);
  f32 entropy = 0.0f;

  for (uint i = 0; i < num_unique; i++) {
    uint count = 0;
    // Compter le nombre d'occurrences pour la classe uniques[i]
    for (uint j = 0; j < Y->rows; j++) {
      if (table_get(Y, j, 0) == uniques[i])
        count++;
    }
    // Calcul de la probabilite p_i
    f32 p = (f32)count / Y->rows;
    if (p > 0.0f) {
      entropy -= p * log2f(p);
    }
  }
  free(uniques);
  return entropy;
}

// Calcul du Gain d'Information : IG(S, A) = H(S) - Somme((|S_v| / |S|) *
// H(S_v))
f32 calc_information_gain(const Table *X, const Table *Y, uint feature_idx,
                          f32 current_entropy) {
  uint num_unique;
  f32 *uniques = get_unique_values(X, feature_idx, &num_unique);

  f32 weighted_entropy = 0.0f;
  for (uint i = 0; i < num_unique; i++) {
    Table sub_Y;
    Table sub_X = table_filter(X, Y, feature_idx, uniques[i], &sub_Y);

    // Poids du sous-ensemble: |S_v| / |S|
    f32 weight = (f32)sub_Y.rows / Y->rows;
    weighted_entropy += weight * calc_entropy(&sub_Y);

    free_table(&sub_X);
    free_table(&sub_Y);
  }
  free(uniques);

  return current_entropy - weighted_entropy;
}

// Construction recursive de l'Arbre de Decision (ID3)
TreeNode *build_tree(const Table *X, const Table *Y, bool *used_features) {
  TreeNode *node = malloc(sizeof(TreeNode));
  node->is_leaf = false;
  node->num_children = 0;
  node->children = NULL;
  node->split_values = NULL;

  uint num_unique_y;
  f32 *uniques_y = get_unique_values(Y, 0, &num_unique_y);

  // Cas de base 1 : Si tous les exemples ont la meme classe
  if (num_unique_y == 1) {
    node->is_leaf = true;
    node->label = uniques_y[0];
    free(uniques_y);
    return node;
  }

  // Trouver la classe majoritaire au cas ou on doit s'arreter
  f32 most_common_y = uniques_y[0];
  uint max_count = 0;
  for (uint i = 0; i < num_unique_y; i++) {
    uint count = 0;
    for (uint j = 0; j < Y->rows; j++) {
      if (table_get(Y, j, 0) == uniques_y[i])
        count++;
    }
    if (count > max_count) {
      max_count = count;
      most_common_y = uniques_y[i];
    }
  }
  free(uniques_y);

  // Cas de base 2 : Si tous les attributs ont ete utilises
  bool no_features = true;
  for (uint i = 0; i < X->cols; i++) {
    if (!used_features[i]) {
      no_features = false;
      break;
    }
  }

  if (no_features) {
    node->is_leaf = true;
    node->label = most_common_y;
    return node;
  }

  // Etape 1 : Trouver le meilleur attribut pour la separation
  f32 current_entropy = calc_entropy(Y);
  f32 best_gain = -1.0f;
  int best_feature = -1;

  for (uint i = 0; i < X->cols; i++) {
    if (!used_features[i]) {
      f32 gain = calc_information_gain(X, Y, i, current_entropy);
      if (gain > best_gain) {
        best_gain = gain;
        best_feature = i;
      }
    }
  }

  // Si on ne peut plus gagner d'information, on cree une feuille
  if (best_feature == -1 || best_gain <= 0.0f) {
    node->is_leaf = true;
    node->label = most_common_y;
    return node;
  }

  // Etape 2 : Separer les donnees selon le meilleur attribut trouve
  node->feature_idx = best_feature;
  used_features[best_feature] = true;

  uint num_unique_x;
  f32 *uniques_x = get_unique_values(X, best_feature, &num_unique_x);
  node->num_children = num_unique_x;
  node->split_values = uniques_x;
  node->children = malloc(num_unique_x * sizeof(TreeNode *));

  // Etape 3 : Creer les sous-arbres (branches)
  for (uint i = 0; i < num_unique_x; i++) {
    Table sub_Y;
    Table sub_X = table_filter(X, Y, best_feature, uniques_x[i], &sub_Y);

    if (sub_X.rows == 0) {
      // Cas de base 3 : Sous-ensemble vide
      TreeNode *leaf = malloc(sizeof(TreeNode));
      leaf->is_leaf = true;
      leaf->label = most_common_y;
      leaf->num_children = 0;
      leaf->children = NULL;
      leaf->split_values = NULL;
      node->children[i] = leaf;
    } else {
      bool *used_features_copy = malloc(X->cols * sizeof(bool));
      for (uint k = 0; k < X->cols; k++)
        used_features_copy[k] = used_features[k];

      node->children[i] = build_tree(&sub_X, &sub_Y, used_features_copy);
      free(used_features_copy);
    }

    free_table(&sub_X);
    free_table(&sub_Y);
  }

  used_features[best_feature] = false;
  return node;
}

// Prediction d'une nouvelle donnee (une ligne de X) en parcourant l'arbre (ID3)
f32 predict(TreeNode *node, const Table *X, uint row) {
  if (node->is_leaf)
    return node->label;

  f32 val = table_get(X, row, node->feature_idx);

  for (uint i = 0; i < node->num_children; i++) {
    if (node->split_values[i] == val) {
      return predict(node->children[i], X, row);
    }
  }

  // Valeur non vue: premier enfant par defaut
  if (node->num_children > 0) {
    return predict(node->children[0], X, row);
  }
  return 0.0f;
}

// Libere la memoire allouee pour un arbre
void free_tree(TreeNode *node) {
  if (!node)
    return;
  for (uint i = 0; i < node->num_children; i++) {
    free_tree(node->children[i]);
  }
  if (node->children)
    free(node->children);
  if (node->split_values)
    free(node->split_values);
  free(node);
}

// --- Nouvelles fonctions pour la Foret Aleatoire (Random Forest) ---

// Etape 1 (Bagging): Creer un nouvel echantillon par tirage avec remise (Bootstrap)
void table_bootstrap(const Table *X, const Table *Y, Table *out_X, Table *out_Y) {
  *out_X = init_table(X->rows, X->cols);
  *out_Y = init_table(Y->rows, Y->cols);

  for (uint i = 0; i < X->rows; i++) {
    // Tirage aleatoire d'un index entre 0 et N-1
    uint rand_idx = rand() % X->rows;
    
    // Copier la ligne rand_idx dans la nouvelle table
    for (uint j = 0; j < X->cols; j++) {
      table_set(out_X, i, j, table_get(X, rand_idx, j));
    }
    table_set(out_Y, i, 0, table_get(Y, rand_idx, 0));
  }
}

// Construction de la foret
RandomForest build_random_forest(const Table *X, const Table *Y, uint num_trees) {
  RandomForest rf;
  rf.num_trees = num_trees;
  rf.trees = malloc(num_trees * sizeof(TreeNode *));

  for (uint i = 0; i < num_trees; i++) {
    Table boot_X, boot_Y;
    // Tirage avec remise d'un nouvel ensemble de donnees
    table_bootstrap(X, Y, &boot_X, &boot_Y);

    bool *used_features = malloc(boot_X.cols * sizeof(bool));
    for (uint j = 0; j < boot_X.cols; j++)
      used_features[j] = false;

    // Entrainer un arbre sur cet echantillon bootstrap
    rf.trees[i] = build_tree(&boot_X, &boot_Y, used_features);

    free(used_features);
    free_table(&boot_X);
    free_table(&boot_Y);
  }

  return rf;
}

// Etape 2 (Aggregation): Prediction par vote majoritaire de la foret
f32 random_forest_predict(const RandomForest *rf, const Table *X, uint row) {
  // Pour stocker les votes des arbres (on suppose des classes de 0 a 9)
  int votes[10] = {0};

  for (uint i = 0; i < rf->num_trees; i++) {
    f32 pred = predict(rf->trees[i], X, row);
    int class_idx = (int)pred;
    if (class_idx >= 0 && class_idx < 10) {
      votes[class_idx]++;
    }
  }

  // Trouver la classe avec le plus de votes
  int best_class = -1;
  int max_votes = -1;
  for (int i = 0; i < 10; i++) {
    if (votes[i] > max_votes) {
      max_votes = votes[i];
      best_class = i;
    }
  }

  return (f32)best_class;
}

// Prediction sur l'ensemble du dataset
Table random_forest_batch_predict(const RandomForest *rf, const Table *X) {
  Table res = init_table(X->rows, 1);
  for (uint i = 0; i < X->rows; i++) {
    table_set(&res, i, 0, random_forest_predict(rf, X, i));
  }
  return res;
}

// Libere la memoire de toute la foret
void free_random_forest(RandomForest *rf) {
  for (uint i = 0; i < rf->num_trees; i++) {
    free_tree(rf->trees[i]);
  }
  free(rf->trees);
}

int main() {
  // Initialisation du generateur de nombres aleatoires
  srand((unsigned int)time(NULL));

  printf("--- Foret Aleatoire (Bootstrap Aggregating) ---\n");

  // Chargement du dataset
  Table data = table_load_csv("datasets/play_tennis.csv", 1);

  // Separation de X (donnees) et Y (cibles/labels)
  Table X = table_extract_columns(&data, 0, data.cols - 1);
  Table Y = table_extract_column(&data, data.cols - 1);

  uint NUM_TREES = 5;
  printf("Entrainement de %u arbres de decision avec Bootstrap...\n", NUM_TREES);
  
  // Entrainement du modele
  RandomForest rf = build_random_forest(&X, &Y, NUM_TREES);

  // Nettoyage de la memoire
  free_table(&X);
  free_table(&Y);
  free_table(&data);
  free_random_forest(&rf);

  return 0;
}
