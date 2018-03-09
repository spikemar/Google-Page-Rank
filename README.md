# Google_Page_Rank
Implmentation of Google Page Rank Algorithm

Serial code build:
gcc gp_rank.c -o gp_rank

Loop Unrolling code build:
gcc gp_rank_unrolling.c -o gp_rank_unrolling

Pthread code build:
gcc -lpthread gp_rank_pthread.c -o gp_rank_pthread

OpenMP code build:
gcc -fopenmp gp_rank_pthread.c -o gp_rank_pthread

CUDA code build:
nvcc -gencode arch=compute_20,code=sm_20 gp_rank_cuda.cu -o gp_rank_cuda


Usage:
./gp_rank <Num_of_nodes> <graph_file>

More real world graphs can be obtained from www.networkrepository.com
