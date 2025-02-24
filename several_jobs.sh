#!/bin/bash

file = "jobScript.sb"
x = 4
for i in $(seq 1 $x); do
sbash $file
done 
