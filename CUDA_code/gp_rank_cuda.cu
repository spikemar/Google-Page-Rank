#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <cuda.h>

#define GIG 1000000000
#define CPG 2.4           // Cycles per GHz -- Adjust to your computer
#define GPU_BLOCK_SIZE 128

typedef float pr_type_t;

typedef struct ad_vert {
	long vertex_num;
	struct ad_vert *next;
}adj_vert_t;

typedef struct {
	pr_type_t curr_page_rank;
	pr_type_t next_page_rank;
	long num_adj_nodes;
	adj_vert_t *last_node_addr;
	void *next;
}vertex_t;

typedef struct {
	long edge_index;
	char is_leaf;
}compact_adj_node_t;

typedef struct {
	pr_type_t next;
	pr_type_t curr;
}p_rank_struct_t;

pr_type_t epsilon;
pr_type_t rand_hop = 0.15;
__device__ pr_type_t d_rand_hop = 0.15;

#define GRAPH_FILE_SEPERATOR " ,;"
#define MAX_LINE_LEN 100
#define RAND_HOP_LIKELIHOOD(r_hop_prob, nvert) ((r_hop_prob) / (nvert))
#define TRAV_LIKELIHOOD(r_hop_prob, p, index, num_adj_nodes) ((1 - (r_hop_prob)) * (p)[index].curr / num_adj_nodes)
#define TRAV_LIKELIHOOD_LEAF(r_hop_prob, p, index, num_vertices) ((1 - (r_hop_prob)) * (p)[index].curr / (num_vertices - 1))

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

long string_to_long(char *str)
{
	long val;
	char *endptr;
	errno = 0;
    val = strtol(str, &endptr, 10);
	if((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN)) || (errno != 0 && val == 0) || (endptr == str)) 
	{
		perror("Error while converting string to long value");
		val = -1;
	}
	return val;
}

void initialize_vertices(vertex_t *g, long num_vertices)
{
	long i;
	for(i = 0;i < num_vertices;i++)
	{
		g[i].curr_page_rank = 1 / (pr_type_t)num_vertices;
		g[i].next_page_rank = RAND_HOP_LIKELIHOOD(rand_hop, num_vertices);
		g[i].num_adj_nodes = 0;
		g[i].last_node_addr = NULL;
		g[i].next = NULL;
	}
}

int append_node(vertex_t *g, long parent_vertex, long child_vertex, long num_verts, long *num_edges)
{
	if(parent_vertex >= num_verts || child_vertex >= num_verts)
	{
		printf("Invalid arguments\n");
		return -1;
	}
	adj_vert_t *ptr = (adj_vert_t *)malloc(sizeof(adj_vert_t));
	ptr->vertex_num = child_vertex;
	ptr->next = NULL;
	if(g[parent_vertex].next == NULL)
	{
		g[parent_vertex].next = ptr;
		g[parent_vertex].last_node_addr = ptr;
	}
	else
	{
		g[parent_vertex].last_node_addr->next = ptr;
		g[parent_vertex].last_node_addr = ptr;
	}
	g[parent_vertex].num_adj_nodes++;
	(*num_edges)++;
	return 0;
}

void calc_bfs_pg_rank_serial(compact_adj_node_t *vertex_array,long *edge_array,char *frontier_array,char *visited_array,p_rank_struct_t *p_rank_array,long num_vertices,long num_edges,long *num_front,long i)
{
	long j, term_ind;
	pr_type_t p_rank_val;
	if(frontier_array[i])
	{
		frontier_array[i] = 0;
		(*num_front)--;
		visited_array[i] = 1;
		term_ind = (i == num_vertices - 1) ? num_edges : vertex_array[i + 1].edge_index;
		p_rank_val = TRAV_LIKELIHOOD(rand_hop,p_rank_array,i,(term_ind - vertex_array[i].edge_index));
		if(!vertex_array[i].is_leaf)
		{
			for(j = vertex_array[i].edge_index;j < term_ind;j++)
			{
				p_rank_array[edge_array[j]].next += p_rank_val;
				if(!visited_array[edge_array[j]])
				{
					if(!frontier_array[edge_array[j]])
					{
						(*num_front)++;
						frontier_array[edge_array[j]] = 1;
					}
				}
			}
		}
		else
		{
			for(j = 0;j < num_vertices && j != i;j++)
				p_rank_array[j].next += TRAV_LIKELIHOOD_LEAF(rand_hop,p_rank_array,i,num_vertices);
		}	
	}
}

__global__ void calc_bfs_pg_rank_cuda(compact_adj_node_t *vertex_array,long *edge_array,p_rank_struct_t *p_rank_array,long num_vertices,long num_edges)
{
	long i, j, term_ind;
	pr_type_t p_rank_val;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < num_vertices)
	{
		term_ind = (i == num_vertices - 1) ? num_edges : vertex_array[i + 1].edge_index;
		if(!vertex_array[i].is_leaf)
		{
			p_rank_val = TRAV_LIKELIHOOD(d_rand_hop,p_rank_array,i,(term_ind - vertex_array[i].edge_index));
			for(j = vertex_array[i].edge_index;j < term_ind;j++)
				atomicAdd(&p_rank_array[edge_array[j]].next,p_rank_val);
		}
		else
		{
			p_rank_val = TRAV_LIKELIHOOD_LEAF(d_rand_hop,p_rank_array,i,num_vertices);
			for(j = 0;j < num_vertices;j++)
			{
				if(j != i)
					atomicAdd(&p_rank_array[j].next,p_rank_val);
			}
		}
	}	
}

__global__ void update_pr(p_rank_struct_t *p_rank_array, pr_type_t *pr_diff_max, long num_vertices)
{
	long i;
	pr_type_t curr_diff;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < num_vertices)
	{
		curr_diff = fabsf(p_rank_array[i].next - p_rank_array[i].curr);
		atomicAdd(pr_diff_max,curr_diff);	
		p_rank_array[i].curr = p_rank_array[i].next;
		p_rank_array[i].next = RAND_HOP_LIKELIHOOD(d_rand_hop,num_vertices);
	}
}

void print_converged_pr_vals(p_rank_struct_t *p_rank, long num_vertices)
{
	long i;
	pr_type_t sum=0;
	for(i = 0;i < num_vertices;i++){
		printf("Converged page rank for node %lu : %.10f\n",i,p_rank[i].curr);
		sum += p_rank[i].curr;
	}
	printf("Sum is %f\n",sum);
}

int main(int argc, char *argv[])
{
	cudaEvent_t start, stop;
	float elapsed_gpu;

	long i,j;
	FILE *file;
	char *token1, *token2;
	char line[MAX_LINE_LEN];
	pr_type_t *d_pr_diff;
	pr_type_t pr_diff;
	long num_vertices = 0;
	long pnode, cnode;
	long iterations=0;
	vertex_t *graph;
	compact_adj_node_t *vertex_array, *d_vertex_array;
	long *edge_array, *d_edge_array;
	p_rank_struct_t *p_rank_array, *d_p_rank_array;
	long num_edges = 0;
	adj_vert_t *avert;
	struct timespec time_diff;
	struct timespec diff(struct timespec start, struct timespec end);
	struct timespec time1, time2;

	if(argc != 3)
		return -1;
	num_vertices = string_to_long(argv[1]);
	if(num_vertices < 0)
		return -1;
	graph = (vertex_t *)malloc(num_vertices * sizeof(vertex_t));
	epsilon =(pr_type_t) 0.00001;
	if(!graph)
		return -1;
	initialize_vertices(graph, num_vertices);
	file = fopen(argv[2],"r");
	if(file)
	{
		while (fgets(line, sizeof(line), file))
		{
			token1 = strtok (line,GRAPH_FILE_SEPERATOR);
			token2 = strtok(NULL,GRAPH_FILE_SEPERATOR);
			if(token1 == NULL || token2 == NULL || strtok(NULL,GRAPH_FILE_SEPERATOR) != NULL)
				return -1;
			pnode = string_to_long(token1);
			cnode = string_to_long(token2);
			if(pnode < 0 || cnode < 0)
				return -1;
			if(append_node(graph,pnode,cnode,num_vertices,&num_edges))
				return -1;
		}
	}
	else
		return -1;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);
	
	//Compact Adjacency list
	vertex_array = (compact_adj_node_t *)calloc(num_vertices, sizeof(compact_adj_node_t));
	edge_array = (long *)calloc(num_edges, sizeof(long));
	p_rank_array = (p_rank_struct_t *)malloc(num_vertices * sizeof(p_rank_struct_t));
	if(cudaMalloc(&d_vertex_array,num_vertices * sizeof(compact_adj_node_t)) != cudaSuccess)
	{
		printf("Error in cudaMalloc\n");
		return -2;
	}
	if(cudaMalloc(&d_edge_array,num_edges * sizeof(long)) != cudaSuccess)
	{
		printf("Error in cudaMalloc\n");
		return -2;
	}
	if(cudaMalloc(&d_p_rank_array,num_vertices * sizeof(p_rank_struct_t)) != cudaSuccess)
	{
		printf("Error in cudaMalloc\n");
		return -2;
	}
	for(i = 0,j = 0;i < num_vertices;i++)
	{
		//Initialize Page Rank values
		p_rank_array[i].next = RAND_HOP_LIKELIHOOD(rand_hop,num_vertices);
		p_rank_array[i].curr = 1 / (pr_type_t)num_vertices;

		vertex_array[i].edge_index = j;
		for(avert = (adj_vert_t *)graph[i].next;avert != NULL;avert = avert->next)
			edge_array[j++] = avert->vertex_num;
		if(vertex_array[i].edge_index - j == 0)
			vertex_array[i].is_leaf = 1;
	}
	if(cudaMemcpy(d_vertex_array,vertex_array,num_vertices * sizeof(compact_adj_node_t),cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Error in cudaMemcpy\n");
		return -2;
	}
	if(cudaMemcpy(d_edge_array,edge_array,num_edges * sizeof(long),cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Error in cudaMemcpy\n");
		return -2;
	}
	if(cudaMemcpy(d_p_rank_array,p_rank_array,num_vertices * sizeof(p_rank_struct_t),cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Error in cudaMemcpy\n");
		return -2;
	}

	printf("Graph parsing successful\n");
	
	if(cudaMallocHost(&d_pr_diff,sizeof(pr_type_t)) != cudaSuccess)
	{
		printf("Error in cudaMalloc\n");
		return -2;
	}

	do
	{
		pr_diff = 0;
		if(cudaMemcpy(d_pr_diff,&pr_diff,sizeof(pr_type_t),cudaMemcpyHostToDevice) != cudaSuccess)
		{
			printf("Error in cudaMemcpy\n");
			return -2;
		}
		calc_bfs_pg_rank_cuda<<<(num_vertices / GPU_BLOCK_SIZE) + 1, GPU_BLOCK_SIZE>>>(d_vertex_array,d_edge_array,d_p_rank_array,num_vertices,num_edges);
		cudaDeviceSynchronize();
		update_pr<<<(num_vertices / GPU_BLOCK_SIZE) + 1, GPU_BLOCK_SIZE>>>(d_p_rank_array, d_pr_diff, num_vertices);
		cudaDeviceSynchronize();
		if(cudaMemcpy(&pr_diff,d_pr_diff,sizeof(pr_type_t),cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			printf("Error in cudaMemcpy\n");
			return -2;
		}
		
	}while(pr_diff > epsilon);
	if(cudaMemcpy(p_rank_array,d_p_rank_array,num_vertices * sizeof(p_rank_struct_t),cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Error in cudaMemcpy\n");
		return -2;
	}
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//print_converged_pr_vals(p_rank_array, num_vertices);

	free(vertex_array);
	free(edge_array);
	free(p_rank_array);

	cudaFree(vertex_array);
	cudaFree(edge_array);
	cudaFreeHost(p_rank_array);
	return 0;
}
