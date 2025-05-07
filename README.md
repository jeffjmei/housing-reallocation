# Housing Reallocation via Network Analysis

This project implements the algorithm described in J.W. Wright's 1975 paper, *Reallocation of Housing by Use of Network Analysis*. It identifies optimal reallocation cycles for housing units based on tenant preferences using network flow techniques.

## Features
- Constructs a directed graph of tenant move requests
- Applies Floyd-Warshall to identify shortest cycles
- Extracts valid exchange circuits with bottleneck demand
- Updates allocation matrix after each cycle

# References
Wright, J. W. (1975). Reallocation of Housing by use of Network Analysis. Journal of the Operational Research Society, 26(2), 253–258. https://doi.org/10.1057/jors.1975.59
