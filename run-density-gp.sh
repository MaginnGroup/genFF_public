#!/bin/bash

for n in 14 32 50 170 125 143a 134a 
do
    mkdir R${n}-densitygp
    cd R${n}-densitygp
    # Copy the run.sh and SGE submission scripts for the equilibration
    sed "s/XXXX/${n}/g" ../gp-density.py > ./gp-density.py
    python gp-density.py
    cd ../
done
