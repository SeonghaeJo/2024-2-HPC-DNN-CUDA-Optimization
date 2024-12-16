#!/bin/bash

: ${NODES:=1}

salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 1 \
		--oversubscribe -quiet \
		ncu -o ./ncu --target-processes all --set full --force-overwrite true ./main -n 16384 -w $@