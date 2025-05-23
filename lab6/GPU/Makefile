INFO = -Minfo=all
LIBS = -lboost_program_options
GPU = -acc -gpu=cc70
OPT = -fast
CXX = pgc++
STD = -std=c++14

all: heat

heat:
	$(CXX) $(STD) $(OPT) $(GPU) $(INFO) main.cpp -o $@ $(LIBS)

clean:
	rm -f heat
