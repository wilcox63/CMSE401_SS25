all: revGOL revGOL-mpi gol

MCC:= mpicc

CC:=gcc

EXT:=c

FLAGS:= -lm -lpng



reverseGOL.o: reverseGOL.$(EXT)

	$(CC) -c reverseGOL.$(EXT)



png_util.o: png_util.c

	$(CC) -c png_util.c



revGOL: reverseGOL.o png_util.o

	$(CC) -o revGOL reverseGOL.o png_util.o $(FLAGS)



gameoflife.o: gameoflife.$(EXT)

	$(CC) -c gameoflife.$(EXT)



gol: gameoflife.o png_util.o

	$(CC) -o gol gameoflife.o png_util.o $(FLAGS)



reverseGOL-mpi.o: reverseGOL-mpi.$(EXT)

	$(MCC) -c reverseGOL-mpi.c



revGOL-mpi: reverseGOL-mpi.o png_util.o

	$(MCC) -o revGOL-mpi reverseGOL-mpi.o png_util.o $(FLAGS)



test: revGOL data.txt

	./revGOL data.txt

	./revGOL-mpi data.txt



clean:

	rm *.o

	rm gol

	rm revGOL

	rm revGOL-mpi
