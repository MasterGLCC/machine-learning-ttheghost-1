#pragma once

typedef unsigned int uint;
typedef float f32;

// Structure de base pour stocker une matrice de donnees (lignes x colonnes)
typedef struct {
  f32 *data;
  uint rows;
  uint cols;
} Table;

// Allocation d'une table de taille rows x cols initialisee a 0
Table init_table(uint rows, uint cols);

// Allocation d'une table de taille rows x cols initialisee avec la valeur val
Table init_table_with(uint rows, uint cols, f32 val);

// Libere la memoire allouee pour une table
void free_table(Table *tab);

// Acces a l'element a la ligne row et colonne col : t_ij = data[row * cols + col]
static inline f32 table_get(const Table *t, uint row, uint col) {
  return t->data[row * t->cols + col];
}

// Modifie l'element a la ligne row et colonne col : t_ij = val
static inline void table_set(Table *t, uint row, uint col, f32 val) {
  t->data[row * t->cols + col] = val;
}

// Acces unidimensionnel (lineaire) a l'element a l'index idx
static inline f32 table_flat_get(const Table *t, uint idx) {
  return t->data[idx];
}

// Modifie l'element a l'index lineaire idx : data[idx] = val
static inline void table_flat_set(Table *t, uint idx, f32 val) {
  t->data[idx] = val;
}

// Cree une copie independante de la table source
Table table_copy(const Table *src);

// Affiche les dimensions de la table : rows x cols
void table_print_shape(const Table *t, const char *name);

// Affiche les n premieres lignes de la table
void table_print_head(const Table *t, uint n, const char *name);

// Extraction de sous-ensembles de donnees
// Extrait une colonne specifique sous forme de vecteur colonne (dimension rows x 1)
Table table_extract_column(const Table *t, uint col_idx);
// Extrait une plage de colonnes [col_start, col_end[
Table table_extract_columns(const Table *t, uint col_start, uint col_end);
// Extrait une ligne specifique sous forme de vecteur ligne (dimension 1 x cols)
Table table_extract_row(const Table *t, uint row_idx);
// Extrait une plage de lignes [row_start, row_end[
Table table_extract_rows(const Table *t, uint row_start, uint row_end);

// Manipulation de lignes
// Echange le contenu de deux lignes de la table
void table_rows_swap(Table *t, uint row_a, uint row_b);
// Melange de maniere synchronisee les lignes des tables X et y
void table_shuffle_together(Table *X, Table *y);
// Combine toutes les lignes de la table source sauf celles dans la plage [start_idx, end_idx[
Table table_combine_except(const Table *src, uint start_idx, uint end_idx);

// --- Fonctions statistiques ---

// Moyenne par colonne : μⱼ = (1/n) Σᵢ xᵢⱼ
Table table_mean_axis0(const Table *X);

// Moyenne par ligne : μᵢ = (1/m) Σⱼ xᵢⱼ
Table table_mean_axis1(const Table *X);

// Ecart-type par colonne : σⱼ = √( (1/n) Σᵢ (xᵢⱼ - μⱼ)² )
Table table_stddev_axis0(const Table *X, const Table *mean);

// Ecart-type par ligne : σᵢ = √( (1/m) Σⱼ (xᵢⱼ - μᵢ)² )
Table table_stddev_axis1(const Table *X, const Table *mean);

// Recherche des valeurs minimales par colonne
Table table_min_axis0(const Table *X);
// Recherche des valeurs maximales par colonne
Table table_max_axis0(const Table *X);

// Normalisation z-score : x' = (x - μ) / σ
void table_normlize_zscore_axis0(Table *X, const Table *mean,
                                 const Table *stddev);

// Denormalisation : x = x' · σ + μ
void table_denormalize_zscore_axis0(Table *X, const Table *mean,
                                    const Table *stddev);
                  
// La somme de tous les elements d'une table : S = ∑ t_ij
f32 table_sum(const Table *t);