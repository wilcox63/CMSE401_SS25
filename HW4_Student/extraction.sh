#!/bin/bash --login

best_fitness=0
best_seed=0
best_file=""

# Loop through all result files
for file in result_*.txt; do
  # Extract the random seed from the file
  random_seed=$(head -n 1 "$file" | grep -o "Random Seed = [0-9]* [0-9]* [0-9]* [0-9]*" | cut -d "=" -f 2)
  
  # Get the last fitness value (from the last generation)
  last_line=$(grep "fitness=" "$file" | tail -n 1)
  fitness=$(echo "$last_line" | grep -o "fitness=[0-9]*" | cut -d "=" -f 2)
  
  # Only process if we found a valid fitness value
  if [ ! -z "$fitness" ] && [[ $fitness =~ ^[0-9]+$ ]]; then
    # For genetic algorithms, lower fitness might be better - adjust comparison as needed
    if [ "$best_fitness" -eq 0 ] || [ "$fitness" -lt "$best_fitness" ]; then
      best_fitness=$fitness
      best_seed="$random_seed"
      best_file="$file"
      best_array_id=$(echo "$file" | grep -o "[0-9]*" | head -n 1)
    fi
    echo "File: $file | Random Seed: $random_seed | Final Fitness: $fitness"
  else
    echo "File: $file | Could not find valid fitness value"
  fi
done

echo "============================="
echo "Best fitness of $best_fitness found in file $best_file"

if [ ! -z "$best_file" ]; then
  echo "Saving best result to pp_best.txt"
  cp "$best_file" pp_best.txt
  echo "===== Summary =====" >> pp_best.txt
  echo "Best fitness: $best_fitness" >> pp_best.txt
  echo "Random Seed: $best_seed" >> pp_best.txt
  echo "Array Task ID: $best_array_id" >> pp_best.txt
  echo "Original file: $best_file" >> pp_best.txt
  echo "Saved to pp_best.txt"
else
  echo "No valid result file found, nothing saved."
fi
