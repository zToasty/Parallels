INFO = -Minfo=all
LIBS = -lboost_program_options
HOST = -acc=host
MULT = -acc=multicore
CXX = pgc++
SRC = main.cpp
CXXFLAGS = $(INFO) $(LIBS)

all: serial multi

serial: $(SRC)
	$(CXX) $(HOST) $(CXXFLAGS) -o $@ $^

multi: $(SRC)
	$(CXX) $(MULT) $(CXXFLAGS) -o $@ $^

clean:
	rm -f serial multi
