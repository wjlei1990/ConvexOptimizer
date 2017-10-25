CXX = g++
CXXFLAGS = --std=c++11

all: optim

optim: funcwrapper.o optimizer.o utils.o main.o
	$(CXX) $(CXXFLAGS) -o optim $^

funcwrapper.o: funcwrapper.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^

optimizer.o: optimizer.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^

utils.o: utils.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^

clean:
	rm -f *.o optim
