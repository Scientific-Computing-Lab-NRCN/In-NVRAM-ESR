CFLAGS = $(shell pkg-config --cflags libpmem) -O2
LDFLAGS = $(shell pkg-config --libs libpmem) -O2
CC = gcc
RM = rm

%.o: %.cpp
	$(CC) -c $< -o $@ $(CFLAGS)


hello_libpmem: hello_libpmem.o 
	 $(CC) -o $@ $< $(LDFLAGS)

clean: 
	rm -f *.o
	rm hello_libpmem
