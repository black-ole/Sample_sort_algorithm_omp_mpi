CC = mpic++
CFLAGS = -Wall -g -fopenmp

all: a3

a3: main.o parallelsort.o
	$(CC) $(CFLAGS) -o a3 main.o parallelsort.o
	
main.o: main.cpp psort.h
	$(CC) $(CFLAGS) -c main.cpp
	
parallelsort.o: parallelsort.cpp psort.h
	$(CC) $(CFLAGS) -c parallelsort.cpp
	
clean:
	rm *.o a3