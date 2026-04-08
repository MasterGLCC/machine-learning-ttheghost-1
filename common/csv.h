#pragma once
#include "math.h"

// Charge un fichier CSV dans une Table, skip_header pour ignorer la 1ere ligne
Table table_load_csv(const char *filename, int skip_header);
