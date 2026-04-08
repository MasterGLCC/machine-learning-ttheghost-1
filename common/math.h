#pragma once

typedef unsigned int uint;
typedef float f32;

typedef struct {
  f32 *data;
  uint rows;
  uint cols;
} Table;

/**
 * Initializer un tableux
 */
Table init_table(uint rows, uint cols);

Table init_table_with(uint rows, uint cols, f32 val);

void free_table(Table *tab);

static inline f32 table_get(const Table *t, uint row, uint col) {
  return t->data[row * t->cols + col];
}

static inline void table_set(Table *t, uint row, uint col, f32 val) {
  t->data[row * t->cols + col] = val;
}

static inline f32 table_flat_get(const Table *t, uint idx) {
  return t->data[idx];
}

static inline void table_flat_set(Table *t, uint idx, f32 val) {
  t->data[idx] = val;
}

Table table_copy(const Table *src);

void table_print_shape(const Table *t, const char *name);

void table_print_head(const Table *t, uint n, const char *name);

Table table_extract_column(const Table *t, uint col_idx);

Table table_extract_columns(const Table *t, uint col_start, uint col_end);

Table table_extract_rows(const Table *t, uint row_start, uint row_end);

// Math

Table table_mean_axis0(const Table *X);

Table table_mean_axis1(const Table *X);

Table table_stddev_axis0(const Table *X, const Table *mean);

Table table_stddev_axis1(const Table *X, const Table *mean);

Table table_min_axis0(const Table *X);

Table table_max_axis0(const Table *X);

// x' = (x - μ) / σ
void table_normlize_zscore_axis0(Table *X, const Table *mean,
                                 const Table *stddev);

// x = (x' * σ) + μ
void table_denormalize_zscore_axis0(Table *X, const Table *mean,
                                    const Table *stddev);
