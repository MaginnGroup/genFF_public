#!/bin/bash

for n in 14 32 50 170 125 143a 134a 
do
    mkdir R${n}-vlegp
    cd R${n}-vlegp
    # Copy the run.sh and SGE submission scripts for the equilibration
    sed "s/XXXX/${n}/g" ../gp-vle.py > ./gp-vle.py
    python gp-vle.py
    cd ../
done
