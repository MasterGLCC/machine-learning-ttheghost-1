CC = clang
CFLAGS = -Wall -Wextra -std=c11 -O3 -I.

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
LR_DIR = linear-regression/c
MLR_DIR = multiple-linear-regression/c

COMMON_SRC = $(COMMON_DIR)/math.c $(COMMON_DIR)/csv.c
LR_SRC = $(LR_DIR)/main.c
MLR_SRC = $(MLR_DIR)/main.c

LR_TARGET = linear_regression$(EXT)
MLR_TARGET = multiple_linear_regression$(EXT)

# Default rule
all: $(LR_TARGET) $(MLR_TARGET)

$(LR_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(LR_SRC) -o $(LR_TARGET) $(LDFLAGS)

$(MLR_TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(MLR_SRC) -o $(MLR_TARGET) $(LDFLAGS)

# Run
run: run_lr run_mlr

run_lr: $(LR_TARGET)
	./$(LR_TARGET)

run_mlr: $(MLR_TARGET)
	./$(MLR_TARGET)

clean:
	del /Q *.exe 2>nul
