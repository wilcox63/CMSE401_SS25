#include <stdio.h>
#include <time.h>

long num_steps = 1;
double step;
int main()
{
	FILE *file = fopen("pi_vs_ts.csv","a");
	int i; double x, pi, sum = 0.0;
	step = 1.0/(double)  num_steps;
	for (i=0;i<num_steps;i++)
	{
		x = (i+0.5) * step;
		sum = sum + 4.0/(1.0+x*x);
	}
	pi = step * sum;
	fprintf(file, "%.15f, %.15f\n", step, pi);
	fclose(file);

	printf("Approximation of pi: %.15f\n", pi);
} 
