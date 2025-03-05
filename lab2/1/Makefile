CC = gcc
CFLAGS = -O2 -fopenmp
TARGET = matrix_vector_omp
SRC = main.c
BUILD_DIR = build
OBJ = $(BUILD_DIR)/$(TARGET).o
BIN = $(BUILD_DIR)/$(TARGET)

.SILENT:

all: $(BIN)

$(BUILD_DIR):
	@echo "Creating build directory"
	mkdir -p $(BUILD_DIR)

$(OBJ): $(SRC) | $(BUILD_DIR)
	@echo "Compiling $(SRC) to $(OBJ)"
	$(CC) $(CFLAGS) -c $(SRC) -o $(OBJ)

$(BIN): $(OBJ)
	@echo "Linking $(OBJ) to $(BIN)"
	$(CC) $(CFLAGS) $(OBJ) -o $(BIN)

.PHONY: clean
clean:
	@echo "Cleaning build directory"
	rm -rf $(BUILD_DIR)
