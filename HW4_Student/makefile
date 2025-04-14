all: revGOL revGOL_mpi gol

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



reverseGOL_mpi.o: reverseGOL_mpi.$(EXT)

	$(MCC) -c reverseGOL_mpi.c



revGOL_mpi: reverseGOL_mpi.o png_util.o

	$(MCC) -o revGOL_mpi reverseGOL_mpi.o png_util.o $(FLAGS)



test: revGOL data.txt

	./revGOL data.txt

	./revGOL_mpi data.txt



clean:

	rm *.o

	rm gol

	rm revGOL

	rm revGOL_mpi
