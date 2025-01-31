#!/bin/bash
echo "Enter a number:"
read number

if [ $number -gt 32 ]; then
	echo "It is above freezing!"
else
	echo "It's below freezing, brrrr!"
fi
