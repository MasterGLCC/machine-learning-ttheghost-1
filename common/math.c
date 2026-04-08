#include "math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Table init_table(uint rows, uint cols) {
  Table tab;
  tab.rows = rows;
  tab.cols = cols;
  tab.data = calloc(rows * cols, sizeof(f32));
  if (!tab.data) {
    fprintf(stderr, "FATAL: init_table(%d, %d) - out of memory\n", rows, cols);
    exit(EXIT_FAILURE);
  }
  return tab;
}

Table init_table_with(uint rows, uint cols, f32 val) {
  Table tab;
  tab.rows = rows;
  tab.cols = cols;
  tab.data = malloc(rows * cols * sizeof(f32));
  if (!tab.data) {
    fprintf(stderr, "FATAL: init_table_with(%d, %d, %f) - out of memory\n",
            rows, cols, val);
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < rows * cols; i++) {
    tab.data[i] = val;
  }
  return tab;
}

void free_table(Table *tab) {
  if (tab && tab->data) {
    free(tab->data);
    tab->data = NULL;
  }
  if (tab) {
    tab->rows = 0;
    tab->cols = 0;
  }
}

Table table_copy(const Table *src) {
  Table dst = init_table(src->rows, src->cols);
  memcpy(dst.data, src->data, (size_t)src->rows * src->cols * sizeof(f32));
  return dst;
}

void table_print_shape(const Table *t, const char *name) {
  printf("%s: shape=(%d, %d), total_elements=%d\n", name, t->rows, t->cols,
         t->rows * t->cols);
}

void table_print_head(const Table *t, uint n, const char *name) {
  uint show = (n < t->rows) ? n : t->rows;
  printf("--- %s (%d rows x %d cols, showing %d) ---\n", name, t->rows, t->cols,
         show);
  printf("\n");
  for (uint i = 0; i < show; i++) {
    printf("  [%4d] ", i);
    for (uint j = 0; j < t->cols; j++) {
      printf("%10.4f ", table_get(t, i, j));
    }
    printf("\n");
  }
  if (t->rows > show)
    printf("  ... (%d more rows)\n", t->rows - show);
  printf("\n");
}

Table table_extract_column(const Table *t, uint col_idx) {
  Table col = init_table(t->rows, 1);
  for (uint i = 0; i < t->rows; i++) {
    table_set(&col, i, 0, table_get(t, i, col_idx));
  }
  return col;
}

Table table_extract_columns(const Table *t, uint col_start, uint col_end) {
  Table sub = init_table(t->rows, (uint)(col_end - col_start));
  for (uint i = 0; i < t->rows; i++) {
    for (uint j = col_start; j < col_end; j++) {
      table_set(&sub, i, (uint)(j - col_start), table_get(t, i, j));
    }
  }
  return sub;
}

Table table_extract_rows(const Table *t, uint row_start, uint row_end) {
  uint new_rows = row_end - row_start;
  Table sub = init_table(new_rows, t->cols);
  memcpy(sub.data, t->data + row_start * t->cols,
         (size_t)new_rows * t->cols * sizeof(f32));
  return sub;
}

Table table_mean_axis0(const Table *X) {
  Table mean = init_table(1, X->cols);
  for (uint j = 0; j < X->cols; j++) {
    f32 sum = 0.0f;
    for (uint i = 0; i < X->rows; i++) {
      sum += table_get(X, i, j);
    }
    table_set(&mean, 0, j, sum / X->rows);
  }
  return mean;
}

Table table_mean_axis1(const Table *X) {
  Table mean = init_table(X->rows, 1);
  for (uint i = 0; i < X->rows; i++) {
    f32 sum = 0.0f;
    for (uint j = 0; j < X->cols; j++) {
      sum += table_get(X, i, j);
    }
    table_set(&mean, i, 0, sum / X->cols);
  }
  return mean;
}

Table table_stddev_axis0(const Table *X, const Table *mean) {
  Table sd = init_table(1, X->cols);
  for (uint j = 0; j < X->cols; j++) {
    f32 sum_sq_diff = 0.0f;
    for (uint i = 0; i < X->rows; i++) {
      f32 diff = table_get(X, i, j) - table_get(mean, 0, j);
      sum_sq_diff += diff * diff;
    }
    table_set(&sd, 0, j, sqrtf(sum_sq_diff / X->rows));
  }
  return sd;
}

Table table_stddev_axis1(const Table *X, const Table *mean) {
  Table sd = init_table(X->rows, 1);
  for (uint i = 0; i < X->rows; i++) {
    f32 sum_sq_diff = 0.0f;
    for (uint j = 0; j < X->cols; j++) {
      f32 diff = table_get(X, i, j) - table_get(mean, i, 0);
      sum_sq_diff += diff * diff;
    }
    table_set(&sd, i, 0, sqrtf(sum_sq_diff / X->cols));
  }
  return sd;
}

Table table_min_axis0(const Table *X) {
  Table mn = init_table(1, X->cols);
  for (uint j = 0; j < X->cols; j++) {
    f32 min_val = table_get(X, 0, j);
    for (uint i = 1; i < X->rows; i++) {
      f32 val = table_get(X, i, j);
      if (val < min_val) {
        min_val = val;
      }
    }
    table_set(&mn, 0, j, min_val);
  }
  return mn;
}

Table table_max_axis0(const Table *X) {
  Table mx = init_table(1, X->cols);
  for (uint j = 0; j < X->cols; j++) {
    f32 max_val = table_get(X, 0, j);
    for (uint i = 1; i < X->rows; i++) {
      f32 val = table_get(X, i, j);
      if (val > max_val) {
        max_val = val;
      }
    }
    table_set(&mx, 0, j, max_val);
  }
  return mx;
}

void table_normlize_zscore_axis0(Table *X, const Table *mean,
                                 const Table *stddev) {
  const f32 eps = 1e-8f;
  for (uint j = 0; j < X->cols; j++) {
    f32 mu = mean->data[j];
    f32 sigma = stddev->data[j] + eps;
    for (uint i = 0; i < X->rows; i++) {
      f32 val = table_get(X, i, j);
      table_set(X, i, j, (val - mu) / sigma);
    }
  }
}

void table_denormalize_zscore_axis0(Table *X, const Table *mean,
                                    const Table *stddev) {
  const f32 eps = 1e-8f;
  for (uint j = 0; j < X->cols; j++) {
    f32 mu = mean->data[j];
    f32 sigma = stddev->data[j] + eps;
    for (uint i = 0; i < X->rows; i++) {
      f32 val = table_get(X, i, j);
      table_set(X, i, j, (sigma * val) + mu);
    }
  }
}
