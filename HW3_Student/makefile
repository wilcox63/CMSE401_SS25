all: gol gol_opt gol_cuda 

CC:=nvcc
EXT:=cu
FLAGS:=-lm -lpng16
OPTFLAGS:= -O2  

gameoflife.o: gameoflife.$(EXT)
	$(CC) $(FLAGS) -c gameoflife.$(EXT) 

png_util.o: png_util.$(EXT)
	$(CC) $(FLAGS) -c png_util.$(EXT)

gol: gameoflife.o png_util.o
	$(CC) $(FLAGS) -o gol gameoflife.o png_util.o

gol_opt: gameoflife.o png_util.o
	$(CC) $(FLAGS) $(OPTFLAGS) -o gol_opt gameoflife.o png_util.o

gol_cuda: gameoflife.o png_util.o
	$(CC) $(FLAGS) -o gol_cuda gameoflife.o png_util.o

test: gol gol_opt gol_cuda data.txt
	@echo "Running serial code"
	./gol < data.txt
	@echo "Running serial code with compiler optimizations."
	./gol_opt < data.txt
	@echo "Running CUDA code"
	./gol_cuda < data.txt

test2: gol data.txt
	cuda-memcheck ./gol < data.txt

clean:
	rm *.o
	rm gol 
	rm gol_opt
	rm gol_cuda

