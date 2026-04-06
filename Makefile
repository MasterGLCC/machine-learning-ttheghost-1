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

COMMON_SRC = $(COMMON_DIR)/math.c
LR_SRC = $(LR_DIR)/main.c

TARGET = linear_regression$(EXT)

# Default rule
all: $(TARGET)

$(TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(LR_SRC) -o $(TARGET) $(LDFLAGS)

# Run
run: $(TARGET)
	./$(TARGET)

clean:
	del /Q *.exe 2>nul