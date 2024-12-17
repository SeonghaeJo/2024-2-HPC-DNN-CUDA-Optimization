#!/bin/bash

: ${NODES:=4}

salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 1 \
		--oversubscribe -quiet \
		nsys profile --force-overwrite true -o ./nsys ./main -n 16384 $@