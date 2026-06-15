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
DECISION_TREE_DIR = decision-tree/from-scratch
RANDOM_FOREST_DIR = random-forest/from-scratch
XGBOOST_DIR = xgboost/from-scratch
DBSCAN_DIR = dbscan/from-scratch
NAIVE_BAYES_DIR = naive-bayes/from-scratch
Q_LEARNING_DIR = q-learning/from-scratch
PCA_DIR = pca/from-scratch
PERCEPTRON_DIR = perceptron/from-scratch

COMMON_SRC = $(COMMON_DIR)/math.c $(COMMON_DIR)/csv.c $(COMMON_DIR)/matrix.c
LR_SRC = $(LR_DIR)/main.c
MLR_SRC = $(MLR_DIR)/main.c
POLY_LR_SRC = $(POLY_LR_DIR)/main.c
LOGISTIC_SRC = $(LOGISTIC_DIR)/main.c
KNN_SRC = $(KNN_DIR)/main.c
SVM_SRC = $(SVM_DIR)/main.c
DECISION_TREE_SRC = $(DECISION_TREE_DIR)/main.c
RANDOM_FOREST_SRC = $(RANDOM_FOREST_DIR)/main.c
XGBOOST_SRC = $(XGBOOST_DIR)/main.c
DBSCAN_SRC = $(DBSCAN_DIR)/main.c
NAIVE_BAYES_SRC = $(NAIVE_BAYES_DIR)/main.c
Q_LEARNING_SRC = $(Q_LEARNING_DIR)/main.c
PCA_SRC = $(PCA_DIR)/main.c
PERCEPTRON_SRC = $(PERCEPTRON_DIR)/main.c

LR_TARGET = linear_regression$(EXT)
MLR_TARGET = multiple_linear_regression$(EXT)
POLY_LR_TARGET = poly_linear_regression$(EXT)
LOGISTIC_TARGET = logistic_regression$(EXT)
KNN_TARGET = knn$(EXT)
SVM_TARGET = svm$(EXT)
DECISION_TREE_TARGET = decision_tree$(EXT)
RANDOM_FOREST_TARGET = random_forest$(EXT)
XGBOOST_TARGET = xgboost$(EXT)
DBSCAN_TARGET = dbscan$(EXT)
NAIVE_BAYES_TARGET = naive_bayes$(EXT)
Q_LEARNING_TARGET = q_learning$(EXT)
PCA_TARGET = pca$(EXT)
PERCEPTRON_TARGET = perceptron$(EXT)

# Default rule
all: $(LR_TARGET) $(MLR_TARGET) $(POLY_LR_TARGET) $(LOGISTIC_TARGET) $(KNN_TARGET) $(SVM_TARGET) $(DECISION_TREE_TARGET) $(RANDOM_FOREST_TARGET) $(XGBOOST_TARGET) $(DBSCAN_TARGET) $(NAIVE_BAYES_TARGET) $(Q_LEARNING_TARGET) $(PCA_TARGET) $(PERCEPTRON_TARGET)

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

$(DECISION_TREE_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(DECISION_TREE_SRC) -o $(DECISION_TREE_TARGET) $(LDFLAGS)

$(RANDOM_FOREST_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(RANDOM_FOREST_SRC) -o $(RANDOM_FOREST_TARGET) $(LDFLAGS)

$(XGBOOST_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(XGBOOST_SRC) -o $(XGBOOST_TARGET) $(LDFLAGS)

$(DBSCAN_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(DBSCAN_SRC) -o $(DBSCAN_TARGET) $(LDFLAGS)

$(NAIVE_BAYES_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(NAIVE_BAYES_SRC) -o $(NAIVE_BAYES_TARGET) $(LDFLAGS)

$(Q_LEARNING_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(Q_LEARNING_SRC) -o $(Q_LEARNING_TARGET) $(LDFLAGS)

$(PCA_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(PCA_SRC) -o $(PCA_TARGET) $(LDFLAGS)

$(PERCEPTRON_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(PERCEPTRON_SRC) -o $(PERCEPTRON_TARGET) $(LDFLAGS)

# Run
run: run_lr run_mlr run_plr run_log run_knn run_svm run_dt run_rf run_xgb run_dbscan run_nb run_ql run_pca run_perceptron

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

run_dt: $(DECISION_TREE_TARGET)
	./$(DECISION_TREE_TARGET)

run_rf: $(RANDOM_FOREST_TARGET)
	./$(RANDOM_FOREST_TARGET)

run_xgb: $(XGBOOST_TARGET)
	./$(XGBOOST_TARGET)

run_dbscan: $(DBSCAN_TARGET)
	./$(DBSCAN_TARGET)

run_nb: $(NAIVE_BAYES_TARGET)
	./$(NAIVE_BAYES_TARGET)

run_ql: $(Q_LEARNING_TARGET)
	./$(Q_LEARNING_TARGET)

run_pca: $(PCA_TARGET)
	./$(PCA_TARGET)

run_perceptron: $(PERCEPTRON_TARGET)
	./$(PERCEPTRON_TARGET)

clean:
	del /Q *.exe 2>nul
