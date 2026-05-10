/**
 * Mohammed IFKIRNE
 */

/**
 * Implementation de l'Arbre de Decision (Algorithme ID3)
 *
 * L'algorithme ID3 (Iterative Dichotomiser 3) construit un arbre de decision
 * a partir d'un ensemble de donnees de maniere recursive.
 *
 * Concepts mathematiques cles:
 * 1. Entropie (Entropy) H(S): Mesure l'imprevisibilite ou l'impurete des
 * donnees. H(S) = - Somme(p_i * log2(p_i)) ou p_i est la proportion d'exemples
 * appartenant a la classe i.
 *
 * 2. Gain d'Information (Information Gain) IG(S, A): Mesure la reduction de
 * l'entropie apres avoir separe l'ensemble S selon l'attribut A. IG(S, A) =
 * H(S) - Somme((|S_v| / |S|) * H(S_v)) ou S_v est le sous-ensemble de S pour
 * lequel l'attribut A a la valeur v.
 *
 * On choisit l'attribut qui maximise le Gain d'Information a chaque noeud.
 */

#include "common/csv.h"
#include "common/math.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Structure d'un noeud de l'arbre de decision
typedef struct TreeNode {
  bool is_leaf;      // Vrai si c'est une feuille (decision finale)
  f32 label;         // Classe predite (valide uniquement si is_leaf = true)
  int feature_idx;   // Index de l'attribut utilise pour la separation
  f32 *split_values; // Valeurs possibles de l'attribut (branches)
  struct TreeNode **children; // Sous-arbres correspondant a chaque valeur
  uint num_children;          // Nombre de branches (enfants)
} TreeNode;

// Fonction utilitaire pour filtrer la table selon une valeur specifique d'une
// colonne Retourne un sous-ensemble de X (et remplit out_Y) ou la colonne 'col'
// vaut 'val'
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

  // Etape 1 : Trouver le meilleur attribut pour la separation (celui avec le
  // plus grand Gain d'Information)
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
  used_features[best_feature] =
      true; // Marquer l'attribut comme utilise dans ce chemin

  uint num_unique_x;
  f32 *uniques_x = get_unique_values(X, best_feature, &num_unique_x);
  node->num_children = num_unique_x;
  node->split_values = uniques_x;
  node->children = malloc(num_unique_x * sizeof(TreeNode *));

  // Etape 3 : Creer les sous-arbres (branches) pour chaque valeur de l'attribut
  // choisi
  for (uint i = 0; i < num_unique_x; i++) {
    Table sub_Y;
    Table sub_X = table_filter(X, Y, best_feature, uniques_x[i], &sub_Y);

    if (sub_X.rows == 0) {
      // Cas de base 3 : Sous-ensemble vide, on prend la classe majoritaire du
      // parent
      TreeNode *leaf = malloc(sizeof(TreeNode));
      leaf->is_leaf = true;
      leaf->label = most_common_y;
      leaf->num_children = 0;
      leaf->children = NULL;
      leaf->split_values = NULL;
      node->children[i] = leaf;
    } else {
      // Copier l'etat des attributs pour eviter d'affecter les autres branches
      bool *used_features_copy = malloc(X->cols * sizeof(bool));
      for (uint k = 0; k < X->cols; k++)
        used_features_copy[k] = used_features[k];

      // Appel recursif pour construire le sous-arbre
      node->children[i] = build_tree(&sub_X, &sub_Y, used_features_copy);
      free(used_features_copy);
    }

    free_table(&sub_X);
    free_table(&sub_Y);
  }

  // Backtrack: liberer l'attribut pour le contexte global
  used_features[best_feature] = false;
  return node;
}

// Prediction d'une nouvelle donnee (une ligne de X) en parcourant l'arbre
f32 predict(TreeNode *node, const Table *X, uint row) {
  // Si on atteint une feuille, on retourne sa classe
  if (node->is_leaf)
    return node->label;

  f32 val = table_get(X, row, node->feature_idx);

  // Parcourir les branches pour trouver la valeur correspondante
  for (uint i = 0; i < node->num_children; i++) {
    if (node->split_values[i] == val) {
      return predict(node->children[i], X, row);
    }
  }

  // Si une valeur non vue apparait, par convention on traverse le premier
  // enfant
  if (node->num_children > 0) {
    return predict(node->children[0], X, row);
  }
  return 0.0f;
}

// Prediction sur l'ensemble du dataset
Table batch_predict(TreeNode *root, const Table *X) {
  Table res = init_table(X->rows, 1);
  for (uint i = 0; i < X->rows; i++) {
    table_set(&res, i, 0, predict(root, X, i));
  }
  return res;
}

// Libere la memoire allouee pour l'arbre
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

// Affichage visuel de l'arbre
void print_tree(TreeNode *node, int depth) {
  for (int i = 0; i < depth; i++)
    printf("  ");
  if (node->is_leaf) {
    printf("-> Feuille: classe %.0f\n", node->label);
  } else {
    printf("[Attribut %d]\n", node->feature_idx);
    for (uint i = 0; i < node->num_children; i++) {
      for (int j = 0; j < depth; j++)
        printf("  ");
      printf(" |- Valeur %.0f:\n", node->split_values[i]);
      print_tree(node->children[i], depth + 1);
    }
  }
}

int main() {
  printf("--- Arbre de Decision (Algorithme ID3) ---\n");

  // Chargement du dataset (ici les valeurs categorielles sont encodees en
  // nombres)
  Table data = table_load_csv("datasets/play_tennis.csv", 1);

  // Separation de X (donnees) et Y (cibles/labels)
  // Les 4 premieres colonnes sont les attributs, la derniere est la cible
  Table X = table_extract_columns(&data, 0, data.cols - 1);
  Table Y = table_extract_column(&data, data.cols - 1);

  // Tableau pour suivre quels attributs ont deja ete utilises dans la branche
  bool *used_features = malloc(X.cols * sizeof(bool));
  for (uint i = 0; i < X.cols; i++)
    used_features[i] = false;

  // Entrainement du modele (construction de l'arbre)
  TreeNode *root = build_tree(&X, &Y, used_features);

  printf("\nStructure de l'arbre:\n");
  print_tree(root, 0);

  // Nettoyage de la memoire
  free(used_features);
  free_table(&X);
  free_table(&Y);
  free_table(&data);
  free_tree(root);

  return 0;
}
