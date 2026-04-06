CC = clang
CFLAGS = -Wall -Wextra -std=c11 -O3 -I.

COMMON_DIR = common
LR_DIR = linear-regression/c

COMMON_SRC = $(COMMON_DIR)/math.c
LR_SRC = $(LR_DIR)/main.c

TARGET = linear_regression.exe

# Default rule
all: $(TARGET)

$(TARGET):
	$(CC) $(CFLAGS) $(COMMON_SRC) $(LR_SRC) -o $(TARGET)

# Run
run: $(TARGET)
	./$(TARGET)

clean:
	del /Q *.exe 2>nul