#include "stdio.h"

#include <math.h>

#include <stdlib.h>

static long num_steps = 10000000;



int main(){

    int steps = num_steps;

    unsigned int seed = 1;

    int range=1000;

    double series[num_steps];

    double avg[num_steps-range];





    //Initialize values in list

    series[0]=10.0;

    for (int i=1;i<steps;i++) {

        series[i]=series[i-1]+ ((double) rand_r(&seed))/RAND_MAX-0.5;

    }

    for (int i=0; i<steps-range;i++){

        avg[i]=0;

    }





    //Compute averages

    for (int i=0; i<steps-range; i++){

        for (int j=0; j<=range;j++){

            avg[i]+=series[i+j];

        }

        avg[i]/=(double)range + 1.0;

    }



    //Print elements for comparison

    printf("%f %f\n\n",series[steps-1],avg[steps-range-1]);

}
