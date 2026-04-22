CC = clang
CFLAGS = -Wall -Wextra -std=c11 -O3 -march=native -I.

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    LDFLAGS = -lm
else
    LDFLAGS =
endif

ifeq ($(OS),Windows_NT)
    EXT = .exe
else
    EXT =
endif

COMMON_DIR = common
LR_DIR = linear-regression/from-scratch
MLR_DIR = multiple-linear-regression/from-scratch
POLY_LR_DIR = poly-linear-regression/from-scratch
LOGISTIC_DIR = logistic-regression/from-scratch
KNN_DIR = k-nearest-neighbors/from-scratch
SVM_DIR = support-vector-machines/from-scratch

COMMON_SRC = $(COMMON_DIR)/math.c $(COMMON_DIR)/csv.c $(COMMON_DIR)/matrix.c
LR_SRC = $(LR_DIR)/main.c
MLR_SRC = $(MLR_DIR)/main.c
POLY_LR_SRC = $(POLY_LR_DIR)/main.c
LOGISTIC_SRC = $(LOGISTIC_DIR)/main.c
KNN_SRC = $(KNN_DIR)/main.c
SVM_SRC = $(SVM_DIR)/main.c

LR_TARGET = linear_regression$(EXT)
MLR_TARGET = multiple_linear_regression$(EXT)
POLY_LR_TARGET = poly_linear_regression$(EXT)
LOGISTIC_TARGET = logistic_regression$(EXT)
KNN_TARGET = knn$(EXT)
SVM_TARGET = svm$(EXT)

# Default rule
all: $(LR_TARGET) $(MLR_TARGET) $(POLY_LR_TARGET) $(LOGISTIC_TARGET) $(KNN_TARGET) $(SVM_TARGET)

$(LR_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(LR_SRC) -o $(LR_TARGET) $(LDFLAGS)

$(MLR_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(MLR_SRC) -o $(MLR_TARGET) $(LDFLAGS)

$(POLY_LR_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(POLY_LR_SRC) -o $(POLY_LR_TARGET) $(LDFLAGS)

$(LOGISTIC_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(LOGISTIC_SRC) -o $(LOGISTIC_TARGET) $(LDFLAGS)

$(KNN_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(KNN_SRC) -o $(KNN_TARGET) $(LDFLAGS)

$(SVM_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(SVM_SRC) -o $(SVM_TARGET) $(LDFLAGS)

# Run
run: run_lr run_mlr run_plr run_log run_knn run_svm

run_lr: $(LR_TARGET)
	./$(LR_TARGET)

run_mlr: $(MLR_TARGET)
	./$(MLR_TARGET)

run_plr: $(POLY_LR_TARGET)
	./$(POLY_LR_TARGET)

run_log: $(LOGISTIC_TARGET)
	./$(LOGISTIC_TARGET)

run_knn: $(KNN_TARGET)
	./$(KNN_TARGET)

run_svm: $(SVM_TARGET)
	./$(SVM_TARGET)

clean:
	del /Q *.exe 2>nul
