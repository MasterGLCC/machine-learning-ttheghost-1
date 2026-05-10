/**
 * Mohammed IFKIRNE
 */

/**
 * Implementation de DBSCAN (Density-Based Spatial Clustering of Applications
 * with Noise)
 *
 * Cet algorithme d'apprentissage non supervise (clustering) regroupe les points
 * qui sont proches les uns des autres (haute densite) et marque comme bruit
 * (noise) les points isoles dans des regions de faible densite.
 *
 * Concepts mathematiques cles:
 * 1. Epsilon (eps) : Distance radiale maximale autour d'un point pour definir
 * son voisinage. On utilise la distance euclidienne: d(p, q) = sqrt( Somme((p_i
 * - q_i)^2) )
 *
 * 2. MinPts : Nombre minimum de points requis dans le voisinage epsilon d'un
 * point pour que ce point soit considere comme un point central (Core Point).
 *
 * Types de points:
 * - Core Point : A au moins MinPts points dans son voisinage.
 * - Border Point : N'est pas un Core Point mais est dans le voisinage d'un Core
 * Point.
 * - Noise (Bruit) : N'est ni Core Point ni Border Point.
 */

#include "common/csv.h"
#include "common/math.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define UNVISITED 0
#define VISITED 1
#define NOISE -1
#define UNASSIGNED 0

// Hyperparametres DBSCAN
#define EPSILON 0.5f
#define MIN_PTS 3

// Calcul de la distance euclidienne entre deux lignes d'une table
f32 euclidean_distance(const Table *X, uint row1, uint row2) {
  f32 sum_sq = 0.0f;
  for (uint j = 0; j < X->cols; j++) {
    f32 diff = table_get(X, row1, j) - table_get(X, row2, j);
    sum_sq += diff * diff;
  }
  return sqrtf(sum_sq);
}

// Trouve tous les voisins d'un point dans un rayon Epsilon
// Retourne un tableau d'indices et met a jour count
uint *region_query(const Table *X, uint point_idx, uint *count) {
  // Allocation maximale possible
  uint *neighbors = malloc(X->rows * sizeof(uint));
  *count = 0;

  for (uint i = 0; i < X->rows; i++) {
    if (euclidean_distance(X, point_idx, i) <= EPSILON) {
      neighbors[(*count)++] = i;
    }
  }
  return neighbors;
}

// Fonction principale DBSCAN
void dbscan(const Table *X, int *clusters) {
  int *state = calloc(X->rows, sizeof(int)); // 0 = UNVISITED, 1 = VISITED
  int current_cluster = 0;

  // Initialisation : tous les points sont non assignes
  for (uint i = 0; i < X->rows; i++) {
    clusters[i] = UNASSIGNED;
  }

  for (uint i = 0; i < X->rows; i++) {
    if (state[i] == VISITED)
      continue;

    state[i] = VISITED;
    uint neighbor_count = 0;
    uint *neighbors = region_query(X, i, &neighbor_count);

    if (neighbor_count < MIN_PTS) {
      clusters[i] = NOISE; // Bruit (faible densite)
    } else {
      current_cluster++;
      clusters[i] = current_cluster;

      // Etendre le cluster a partir des voisins
      // On utilise une boucle while car le tableau des voisins peut grandir
      uint seed_idx = 0;
      while (seed_idx < neighbor_count) {
        uint current_p = neighbors[seed_idx];

        if (state[current_p] == UNVISITED) {
          state[current_p] = VISITED;

          uint current_neighbor_count = 0;
          uint *current_neighbors =
              region_query(X, current_p, &current_neighbor_count);

          // Si c'est un point central (Core Point), on ajoute ses voisins
          if (current_neighbor_count >= MIN_PTS) {
            // Reallocation et fusion manuelle pour ajouter les nouveaux voisins
            uint *new_neighbors = malloc(
                (neighbor_count + current_neighbor_count) * sizeof(uint));
            for (uint k = 0; k < neighbor_count; k++)
              new_neighbors[k] = neighbors[k];

            // Ajouter seulement ceux qui n'y sont pas deja
            for (uint k = 0; k < current_neighbor_count; k++) {
              bool exists = false;
              for (uint m = 0; m < neighbor_count; m++) {
                if (neighbors[m] == current_neighbors[k]) {
                  exists = true;
                  break;
                }
              }
              if (!exists) {
                new_neighbors[neighbor_count++] = current_neighbors[k];
              }
            }
            free(neighbors);
            neighbors = new_neighbors;
          }
          free(current_neighbors);
        }

        // Si le point n'appartient a aucun cluster (meme s'il etait marque
        // NOISE)
        if (clusters[current_p] == UNASSIGNED || clusters[current_p] == NOISE) {
          clusters[current_p] = current_cluster;
        }

        seed_idx++;
      }
    }
    free(neighbors);
  }

  free(state);
  printf("\nDBSCAN termine. Nombre de clusters trouves : %d\n",
         current_cluster);
}

int main() {
  printf("--- DBSCAN (Clustering Spatial Base sur la Densite) ---\n");

  // Chargement du dataset (coordonnees 2D sans variable cible)
  Table X = table_load_csv("datasets/dbscan_data.csv", 1);
  printf("Dataset charge : %u points.\n", X.rows);

  // Tableau pour stocker l'ID du cluster de chaque point
  int *clusters = malloc(X.rows * sizeof(int));

  printf("Parametres : Epsilon = %.2f, MinPts = %d\n", EPSILON, MIN_PTS);

  // Execution de l'algorithme
  dbscan(&X, clusters);

  // Affichage des resultats
  printf("\nResultats du Clustering :\n");
  int noise_count = 0;
  for (uint i = 0; i < X.rows; i++) {
    if (clusters[i] == NOISE) {
      noise_count++;
      printf("Point %2u (%.1f, %.1f) : Bruit\n", i, table_get(&X, i, 0),
             table_get(&X, i, 1));
    } else {
      printf("Point %2u (%.1f, %.1f) : Cluster %d\n", i, table_get(&X, i, 0),
             table_get(&X, i, 1), clusters[i]);
    }
  }
  printf("Points classes comme bruit : %d\n", noise_count);

  free(clusters);
  free_table(&X);

  return 0;
}

// --- Affichage du code ---

// --- DBSCAN (Clustering Spatial Base sur la Densite) ---
// Charge 'datasets/dbscan_data.csv': 14 lignes x 2 colonnes
// Dataset charge : 14 points.
// Parametres : Epsilon = 0.50, MinPts = 3

// DBSCAN termine. Nombre de clusters trouves : 2

// Resultats du Clustering :
// Point  0 (1.0, 1.0) : Cluster 1
// Point  1 (1.1, 1.1) : Cluster 1
// Point  2 (0.9, 1.0) : Cluster 1
// Point  3 (1.0, 0.9) : Cluster 1
// Point  4 (0.9, 0.9) : Cluster 1
// Point  5 (1.1, 0.9) : Cluster 1
// Point  6 (5.0, 5.0) : Cluster 2
// Point  7 (5.1, 5.1) : Cluster 2
// Point  8 (4.9, 5.0) : Cluster 2
// Point  9 (5.0, 4.9) : Cluster 2
// Point 10 (4.9, 4.9) : Cluster 2
// Point 11 (5.1, 4.9) : Cluster 2
// Point 12 (8.0, 1.0) : Bruit
// Point 13 (1.0, 8.0) : Bruit
// Points classes comme bruit : 2
