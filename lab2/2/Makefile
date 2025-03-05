CC = gcc
CFLAGS = -O2 -fopenmp
LDFLAGS = -fopenmp -lm
TARGET = integral
SRC = integral.c
BUILD_DIR = build
OBJ = $(BUILD_DIR)/$(TARGET).o
BIN = $(BUILD_DIR)/$(TARGET)

NUM_THREADS ?= 8

.SILENT:

all: $(BIN)

$(BUILD_DIR):
	@echo "Creating build directory"
	mkdir -p $(BUILD_DIR)

$(OBJ): $(SRC) | $(BUILD_DIR)
	@echo "Compiling $(SRC) to $(OBJ) with NUM_THREADS=$(NUM_THREADS)"
	$(CC) $(CFLAGS) -c $(SRC) -DNUM_THREADS=$(NUM_THREADS) -o $(OBJ)

$(BIN): $(OBJ)
	@echo "Linking $(OBJ) to $(BIN)"
	$(CC) $(OBJ) -o $(BIN) $(CFLAGS) $(LDFLAGS)

.PHONY: clean
clean:
	@echo "Cleaning build directory"
	rm -rf $(BUILD_DIR)
	