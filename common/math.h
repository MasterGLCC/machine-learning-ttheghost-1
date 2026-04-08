#pragma once

typedef unsigned int uint;
typedef float f32;

// Structure de base pour stocker une matrice de données (lignes x colonnes)
typedef struct {
  f32 *data;
  uint rows;
  uint cols;
} Table;

// Allocation d'une table initialisée à 0
Table init_table(uint rows, uint cols);

// Allocation d'une table remplie avec la valeur val
Table init_table_with(uint rows, uint cols, f32 val);

// Libère la mémoire d'une table
void free_table(Table *tab);

// Accès à un élément : data[row * cols + col]
static inline f32 table_get(const Table *t, uint row, uint col) {
  return t->data[row * t->cols + col];
}

static inline void table_set(Table *t, uint row, uint col, f32 val) {
  t->data[row * t->cols + col] = val;
}

// Accès linéaire (sans row/col)
static inline f32 table_flat_get(const Table *t, uint idx) {
  return t->data[idx];
}

static inline void table_flat_set(Table *t, uint idx, f32 val) {
  t->data[idx] = val;
}

Table table_copy(const Table *src);

void table_print_shape(const Table *t, const char *name);

void table_print_head(const Table *t, uint n, const char *name);

// Extraction de sous-ensembles
Table table_extract_column(const Table *t, uint col_idx);
Table table_extract_columns(const Table *t, uint col_start, uint col_end);
Table table_extract_rows(const Table *t, uint row_start, uint row_end);

// --- Fonctions statistiques ---

// Moyenne par colonne : μⱼ = (1/n) Σᵢ xᵢⱼ
Table table_mean_axis0(const Table *X);

// Moyenne par ligne : μᵢ = (1/m) Σⱼ xᵢⱼ
Table table_mean_axis1(const Table *X);

// Écart-type par colonne : σⱼ = √( (1/n) Σᵢ (xᵢⱼ - μⱼ)² )
Table table_stddev_axis0(const Table *X, const Table *mean);

// Écart-type par ligne
Table table_stddev_axis1(const Table *X, const Table *mean);

// Min/max par colonne
Table table_min_axis0(const Table *X);
Table table_max_axis0(const Table *X);

// Normalisation z-score : x' = (x - μ) / σ
void table_normlize_zscore_axis0(Table *X, const Table *mean,
                                 const Table *stddev);

// Dénormalisation : x = x' · σ + μ
void table_denormalize_zscore_axis0(Table *X, const Table *mean,
                                    const Table *stddev);