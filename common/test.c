#include "math.h"
#include <stdio.h>

int main() {
    Table t = init_table_with(5, 4, 8.7);
    table_set(&t, 2, 1, -0.6);
    table_print_head(&t, -1, "Test");

    Table mean = table_mean_axis0(&t);
    Table stddev = table_stddev_axis0(&t, &mean);
    table_normlize_zscore_axis0(&t, &mean, &stddev);
    table_print_head(&t, -1, "normlized");
    return 0;
}