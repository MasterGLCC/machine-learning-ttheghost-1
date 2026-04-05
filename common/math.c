#include "math.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Table init_table(uint rows, uint cols) {
    Table tab;
    tab.rows = rows;
    tab.cols = cols;
    tab.data = calloc(rows * cols, sizeof(f32));
    if (!tab.data)
    {
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
    if (!tab.data)
    {
        fprintf(stderr, "FATAL: init_table_with(%d, %d, %f) - out of memory\n", rows, cols, val);
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < rows * cols; i++)
    {
        tab.data[i] = val;
    }
    return tab;
}

void free_table(Table *tab) {
    if (tab && tab->data)
    {
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

void table_print_shape(const Table *t, const char *name);