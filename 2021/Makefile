CC = clang++
CC_FLAGS = -O3 -Rpass=loop-vectorize -std=c++2a -fno-omit-frame-pointer \
  -L /usr/local/lib -lbenchmark -lpthread -march=native

all: day1

clean:
	rm day1

day1: day1.cc day1.input
	$(CC) day1.cc $(CC_FLAGS) -o day1
