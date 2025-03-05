CXX = g++
CXXFLAGS = -std=c++17 -O2
TARGET = SinusArray

ifdef USE_FLOAT
    CXXFLAGS += -DUSE_FLOAT
endif

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp

clean:
	rm -f $(TARGET)