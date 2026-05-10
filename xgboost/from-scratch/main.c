/**
 * Mohammed IFKIRNE
 */

/**
 * Implementation Simplifiee de XGBoost (Gradient Boosting)
 * 
 * Cet algorithme construit une serie d'arbres de regression qui corrigent
 * sequentiellement les erreurs des arbres precedents en optimisant une 
 * fonction objectif regularisee.
 * 
 * Concepts mathematiques cles:
 * 1. Fonction Objectif: Obj = L + Omega (Perte + Regularisation)
 *    L'approximation de Taylor au second ordre de la perte donne:
 *    Obj ≈ Somme(g_i * f_t(x_i) + 1/2 * h_i * f_t(x_i)^2) + Omega(f_t)
 * 
 * 2. Gradients et Hessiens: Pour la classification binaire (Log-Loss) :
 *    - Probabilite: p_i = 1 / (1 + exp(-y_pred_i))
 *    - Gradient (Derivee 1ere) : g_i = p_i - y_i
 *    - Hessien (Derivee 2nde)  : h_i = p_i * (1 - p_i)
 * 
 * 3. Poids d'une Feuille Optimal (w*):
 *    w* = - Somme(g_i) / (Somme(h_i) + lambda)
 * 
 * 4. Gain d'une Separation (Split Gain):
 *    Gain = 1/2 * [ Somme_v (G_v^2 / (H_v + lambda)) - (G^2 / (H + lambda)) ] - gamma
 *    ou G_v et H_v sont les sommes des gradients et hessiens pour la branche v.
 */

#include "common/csv.h"
#include "common/math.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Hyperparametres de XGBoost
#define LAMBDA 1.0f     // Regularisation L2 sur les poids
#define GAMMA 0.0f      // Penalite minimale pour creer une nouvelle branche
#define MAX_DEPTH 3     // Profondeur maximale de l'arbre
#define LEARNING_RATE 0.3f
#define NUM_BOOST_ROUND 10

// Structure d'un noeud d'arbre XGBoost
typedef struct TreeNode {
  bool is_leaf;
  f32 weight;        // Poids de la feuille (w*)
  int feature_idx;
  f32 *split_values;
  struct TreeNode **children;
  uint num_children;
} TreeNode;

// Calcule la fonction Sigmoide
f32 sigmoid(f32 x) {
  return 1.0f / (1.0f + expf(-x));
}

// Filtre la table et les tableaux de gradients/hessiens
Table table_filter_xgb(const Table *X, const f32 *g, const f32 *h, uint col, f32 val, f32 **out_g, f32 **out_h) {
  uint count = 0;
  for (uint i = 0; i < X->rows; i++) {
    if (table_get(X, i, col) == val) count++;
  }

  Table res_X = init_table(count, X->cols);
  *out_g = malloc(count * sizeof(f32));
  *out_h = malloc(count * sizeof(f32));

  uint idx = 0;
  for (uint i = 0; i < X->rows; i++) {
    if (table_get(X, i, col) == val) {
      for (uint j = 0; j < X->cols; j++) {
        table_set(&res_X, idx, j, table_get(X, i, j));
      }
      (*out_g)[idx] = g[i];
      (*out_h)[idx] = h[i];
      idx++;
    }
  }
  return res_X;
}

// Trouve toutes les valeurs uniques presentes dans une colonne
f32 *get_unique_values(const Table *t, uint col, uint *num_unique) {
  f32 *uniques = malloc(t->rows * sizeof(f32));
  uint count = 0;
  for (uint i = 0; i < t->rows; i++) {
    f32 val = table_get(t, i, col);
    bool found = false;
    for (uint j = 0; j < count; j++) {
      if (uniques[j] == val) { found = true; break; }
    }
    if (!found) uniques[count++] = val;
  }
  *num_unique = count;
  return uniques;
}

// Construction recursive de l'arbre XGBoost
TreeNode *build_xgb_tree(const Table *X, const f32 *g, const f32 *h, int depth) {
  TreeNode *node = malloc(sizeof(TreeNode));
  node->is_leaf = false;
  node->num_children = 0;
  node->children = NULL;
  node->split_values = NULL;

  f32 sum_g = 0.0f;
  f32 sum_h = 0.0f;
  for (uint i = 0; i < X->rows; i++) {
    sum_g += g[i];
    sum_h += h[i];
  }

  // Calcul du poids de la feuille actuel : w* = -G / (H + lambda)
  f32 leaf_weight = -sum_g / (sum_h + LAMBDA);

  // Conditions d'arret : Profondeur max ou plus de donnees a separer
  if (depth >= MAX_DEPTH || X->rows <= 1) {
    node->is_leaf = true;
    node->weight = leaf_weight;
    return node;
  }

  f32 best_gain = 0.0f;
  int best_feature = -1;
  
  // Score initial du noeud parent : G^2 / (H + lambda)
  f32 root_score = (sum_g * sum_g) / (sum_h + LAMBDA);

  // Etape 1: Recherche de la meilleure separation pour maximiser le Gain
  for (uint j = 0; j < X->cols; j++) {
    uint num_unique;
    f32 *uniques = get_unique_values(X, j, &num_unique);

    if (num_unique > 1) {
      f32 split_score = 0.0f;
      for (uint v = 0; v < num_unique; v++) {
        f32 G_v = 0.0f, H_v = 0.0f;
        for (uint i = 0; i < X->rows; i++) {
          if (table_get(X, i, j) == uniques[v]) {
            G_v += g[i];
            H_v += h[i];
          }
        }
        split_score += (G_v * G_v) / (H_v + LAMBDA);
      }

      // Equation du Gain de la methode XGBoost
      f32 gain = 0.5f * (split_score - root_score) - GAMMA;

      if (gain > best_gain) {
        best_gain = gain;
        best_feature = j;
      }
    }
    free(uniques);
  }

  // Si aucun gain positif n'est trouve, on s'arrete et cree une feuille
  if (best_feature == -1 || best_gain <= 0.0f) {
    node->is_leaf = true;
    node->weight = leaf_weight;
    return node;
  }

  // Etape 2: Separation et appels recursifs sur le meilleur attribut
  node->feature_idx = best_feature;
  uint num_unique_x;
  f32 *uniques_x = get_unique_values(X, best_feature, &num_unique_x);
  node->num_children = num_unique_x;
  node->split_values = uniques_x;
  node->children = malloc(num_unique_x * sizeof(TreeNode *));

  for (uint i = 0; i < num_unique_x; i++) {
    f32 *sub_g, *sub_h;
    Table sub_X = table_filter_xgb(X, g, h, best_feature, uniques_x[i], &sub_g, &sub_h);

    if (sub_X.rows == 0) {
      TreeNode *leaf = malloc(sizeof(TreeNode));
      leaf->is_leaf = true;
      leaf->weight = leaf_weight; // Poids parent par defaut
      node->children[i] = leaf;
    } else {
      node->children[i] = build_xgb_tree(&sub_X, sub_g, sub_h, depth + 1);
    }

    free_table(&sub_X);
    free(sub_g);
    free(sub_h);
  }

  return node;
}

// Prediction d'un arbre pour une seule ligne : renvoie le poids de la feuille w*
f32 predict_tree(TreeNode *node, const Table *X, uint row) {
  if (node->is_leaf) return node->weight;

  f32 val = table_get(X, row, node->feature_idx);
  for (uint i = 0; i < node->num_children; i++) {
    if (node->split_values[i] == val) {
      return predict_tree(node->children[i], X, row);
    }
  }
  return 0.0f; // Si valeur non vue
}

// Libere la memoire de l'arbre
void free_tree(TreeNode *node) {
  if (!node) return;
  for (uint i = 0; i < node->num_children; i++) {
    if (node->children && node->children[i]) free_tree(node->children[i]);
  }
  if (node->children) free(node->children);
  if (node->split_values) free(node->split_values);
  free(node);
}

int main() {
  printf("--- XGBoost Simplifie (Gradient Boosting) ---\n");

  Table data = table_load_csv("datasets/play_tennis.csv", 1);
  Table X = table_extract_columns(&data, 0, data.cols - 1);
  Table Y = table_extract_column(&data, data.cols - 1);

  // Initialisation des predictions brutes (Logits) a 0.0f
  f32 *preds = calloc(X.rows, sizeof(f32));
  f32 *g = malloc(X.rows * sizeof(f32));
  f32 *h = malloc(X.rows * sizeof(f32));

  TreeNode *trees[NUM_BOOST_ROUND];

  // Boucle principale de Gradient Boosting
  for (int round = 0; round < NUM_BOOST_ROUND; round++) {
    
    // Etape 1: Calcul des probabilites, gradients (g) et hessiens (h)
    for (uint i = 0; i < X.rows; i++) {
      f32 p = sigmoid(preds[i]);
      f32 y = table_get(&Y, i, 0);
      g[i] = p - y;             // Derivee premiere (Gradient)
      h[i] = p * (1.0f - p);    // Derivee seconde (Hessien)
    }

    // Etape 2: Construction de l'arbre pour minimiser les residus
    trees[round] = build_xgb_tree(&X, g, h, 0);

    // Etape 3: Mise a jour des predictions avec le Taux d'Apprentissage
    for (uint i = 0; i < X.rows; i++) {
      preds[i] += LEARNING_RATE * predict_tree(trees[round], &X, i);
    }
  }

  printf("Entrainement de %d arbres termine.\n", NUM_BOOST_ROUND);

  // Nettoyage de la memoire
  for (int round = 0; round < NUM_BOOST_ROUND; round++) {
    free_tree(trees[round]);
  }
  free(preds);
  free(g);
  free(h);
  free_table(&X);
  free_table(&Y);
  free_table(&data);

  return 0;
}
