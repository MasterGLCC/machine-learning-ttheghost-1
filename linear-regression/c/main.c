#include "common/math.h"
#include <stdio.h>

int main() {
    Table t = init_table_with(3, 4, 1.0f);
    table_print_shape(&t, "t");
    table_print_head(&t, 5, "t");
    free_table(&t);
    return 0;
};