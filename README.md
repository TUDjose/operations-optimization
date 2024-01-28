# MILP model for the repatriation scheduling problem

## Description
This repository contains the code for the implementation of the 
paper "A mixed integer linear programming model and a basic variable neighbourhood search algorithm for the repatriation scheduling problem"
(https://www.sciencedirect.com/science/article/pii/S0957417422002019).

This was implemented as part of the course Operations Optimization, by Group 26:
- José Cunha (5216087)
- Pablo Garcia (5270944)
- Stijn Koelemaij (5089344)

## How to run
To run the solving of the RSP model, run the following command, where the parameters are:
- file: filename of the input file
- u: number of aircraft or aircraft trips
- qc: quarantine capacity
- capacity: aircraft capacity
```bash
python -m run -file 'filename' -u 4 -qc 2500 -capacity 300
```
\
To run the generation of random instances, run the following command, where the parameters are:
- m: number of cities
- n: number of priority groups
- u: number of aircraft or aircraft trips
- fq: ratio of fleet capacity to quarantine capacity
```bash
python -m run --gen -m 5 -n 4 -u 2 -fq 1.2
```
\
To run the sensitivity analysis, run the following command:
```bash
python -m run --sens
```




