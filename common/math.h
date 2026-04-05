#pragma once

typedef unsigned int uint;
typedef float f32;

typedef struct
{
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

static inline float table_get(const Table *t, uint row, uint col) {
  return t->data[row * t->cols + col];
}

static inline float table_set(const Table *t, uint row, uint col, f32 val) {
  t->data[row * t->cols + col] = val;
}

static inline float table_flat_get(const Table *t, int idx) {
    return t->data[idx];
}

static inline void table_flat_set(Table *t, int idx, float val) {
    t->data[idx] = val;
}

Table table_copy(const Table *src);
