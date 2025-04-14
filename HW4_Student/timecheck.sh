#!/bin/bash --login

#SBATCH --job-name=serialROC_benchmark

#SBATCH --time=05:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=2G

#SBATCH --output=revROC_benchmark_%j.out



OUTPUT_FILE="benchmark_results.txt"

BEST_SOLUTION="serial_best.txt"

PROGRAM="./revGOL"

INPUT_FILE="cmse2.txt"

NUM_RUNS=10



echo "Start time: $(date)"



best_fitness=999999

best_seed=0

total_time=0



echo "Seed, Time(s), Fitness" > $OUTPUT_FILE



for SEED in $(seq 1 $NUM_RUNS); do

    echo "Running with SEED=$SEED..."

    

    start_time=$(date +%s.%N)

    OUTPUT=$($PROGRAM $INPUT_FILE $SEED)

    end_time=$(date +%s.%N)

    

    EXEC_TIME=$(echo "$end_time - $start_time" | bc)

    

    FITNESS=$(echo "$OUTPUT" | grep "Result Fitness=" | sed 's/.*Result Fitness=\([0-9]*\).*/\1/')

    

    # Fallback if parsing fails

    if [ -z "$FITNESS" ]; then

        FITNESS=999999

    fi



    echo "$SEED, $EXEC_TIME, $FITNESS" >> $OUTPUT_FILE

    total_time=$(echo "$total_time + $EXEC_TIME" | bc)

    

    if [[ "$FITNESS" =~ ^[0-9]+$ ]] && [ "$FITNESS" -lt "$best_fitness" ]; then

        best_fitness=$FITNESS

        best_seed=$SEED

        echo "$OUTPUT" > $BEST_SOLUTION

    fi



    echo "Completed SEED=$SEED: Time=$EXEC_TIME seconds, Fitness=$FITNESS"

done



avg_time=$(echo "scale=3; $total_time / $NUM_RUNS" | bc)



echo "------------------------------------"

echo "Benchmark Complete!"

echo "------------------------------------"

echo "Average execution time: $avg_time seconds"

echo "Best fitness: $best_fitness (from seed $best_seed)"

echo "Best solution saved to $BEST_SOLUTION"

echo "Full results saved to $OUTPUT_FILE"

echo "End time: $(date)"


