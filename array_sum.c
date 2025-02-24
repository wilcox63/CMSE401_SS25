#include "omp.h"
#include <stdio.h>

int main() {
	double sum = 0.0;

	#pragma omp parallel
	{
	int ID = omp_get_thread_num();
	sum += ID;
	}
printf("The sum: %f\n", sum);
}
