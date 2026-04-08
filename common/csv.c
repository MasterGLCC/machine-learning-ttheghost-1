#define _CRT_SECURE_NO_WARNINGS
#include "csv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Lecture d'un fichier CSV et stockage dans une Table
// On fait 2 passes : la 1ere pour compter lignes/colonnes, la 2eme pour remplir
Table table_load_csv(const char *filename, int skip_header) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "ERREUR: impossible d'ouvrir '%s'\n", filename);
    exit(EXIT_FAILURE);
  }

  char line[8192];
  uint rows = 0, cols = 0;

  // 1ere passe : on compte les lignes et on detecte le nb de colonnes
  while (fgets(line, sizeof(line), fp)) {
    if (skip_header && rows == 0) {
      rows++;
      continue;
    }
    if (cols == 0) {
      cols = 1;
      for (char *p = line; *p; p++) {
        if (*p == ',')
          cols++;
      }
    }
    rows++;
  }
  uint data_rows = skip_header ? rows - 1 : rows;

  // 2eme passe : on remplit la table avec les valeurs
  Table t = init_table(data_rows, cols);
  rewind(fp);

  uint row_idx = 0, line_num = 0;
  while (fgets(line, sizeof(line), fp)) {
    line_num++;
    if (skip_header && line_num == 1)
      continue;

    uint col_idx = 0;
    char *token = strtok(line, ",\n\r");
    while (token && col_idx < cols) {
      table_set(&t, row_idx, col_idx, (f32)atof(token));
      token = strtok(NULL, ",\n\r");
      col_idx++;
    }
    row_idx++;
  }
  fclose(fp);
  printf("Chargé '%s': %d lignes x %d colonnes\n", filename, data_rows, cols);
  return t;
}
