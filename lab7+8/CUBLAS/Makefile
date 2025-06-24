INFO = -Minfo=all
LIBS = -cudalib=cublas -lboost_program_options
GPU = -acc=gpu
OPT = -fast
CXX = pgc++
STD = -std=c++14

all: heat

heat:
	$(CXX) $(STD) $(OPT) $(GPU) $(INFO) main.cpp -o $@ $(LIBS)

clean:
	rm -f heat
