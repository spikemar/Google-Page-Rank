#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define GIG 1000000000
#define CPG 2.4           // Cycles per GHz -- Adjust to your computer
#define NUM_THREADS 500

struct Thread
{
  int tid;
  int start_node, end_node;
  int num_vertices;
}; 

//pthread_t *Threads;
//Thread *Threads_data;

typedef struct ad_vert {
	long vertex_num;
	struct ad_vert *next;
}adj_vert_t;


typedef struct {
	double curr_page_rank;
	double next_page_rank;
	long num_adj_nodes;
	adj_vert_t *last_node_addr;
	void *next;
}vertex_t;

vertex_t *graph;

double epsilon;
double rand_hop = 0.15;
#define GRAPH_FILE_SEPERATOR " ,;"
#define MAX_LINE_LEN 100
#define RAND_HOP_LIKELIHOOD(r_hop_prob, nvert) ((r_hop_prob) / (nvert))
#define TRAV_LIKELIHOOD(r_hop_prob, g, index,num_vertices) ((1 - (r_hop_prob)) * (g)[index].curr_page_rank / (g)[index].num_adj_nodes)
#define TRAV_LIKELIHOOD_LEAF(r_hop_prob, g, index,num_vertices) ((1 - (r_hop_prob)) * (g)[index].curr_page_rank / (num_vertices - 1))



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

//void *PageRank_compute();


void initialize_vertices(vertex_t *g, long num_vertices)
{
	long i;
	for(i = 0;i < num_vertices;i++)
	{
		g[i].curr_page_rank = 1 / (double)num_vertices;
		g[i].next_page_rank = RAND_HOP_LIKELIHOOD(rand_hop, num_vertices);
		g[i].num_adj_nodes = 0;
		g[i].last_node_addr = NULL;
		g[i].next = NULL;
	}
}

int append_node(vertex_t *g, long parent_vertex, long child_vertex, long num_verts)
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
	return 0;
}

void print_converged_pr_vals(vertex_t *g, long num_vertices)
{
	long i;
	double sum=0;  adj_vert_t *ptr = NULL;

	for(i = 0;i < num_vertices;i++){
		printf("Converged page rank for node %lu : %.10f\n",i,g[i].curr_page_rank);
		sum += g[i].curr_page_rank;
	}
	printf("Sum is %f\n",sum);

}

void* parallel_pagerank(void* arg){
  int i,j;
  float value;
  struct Thread *thread_data = (struct Thread *)arg;
  adj_vert_t *ptr = NULL;

  //printf("%d\t",thread_data->tid);
  //printf("%d\t",thread_data->start_node);
  //printf("%d\n",thread_data->end_node);

  //printf("%d\n", thread_data->tid);
  for(i = thread_data->start_node;i < thread_data->end_node;i++)
	{
	  if(graph[i].next == NULL)
	    {
	      for(j = 0;j < thread_data->num_vertices;j++)
		{
		  if(j != i)
		    graph[j].next_page_rank += TRAV_LIKELIHOOD_LEAF(rand_hop,graph, i, thread_data->num_vertices);
		}
	    }
	  else
	    value = TRAV_LIKELIHOOD(rand_hop, graph, i, num_vertices);
	  for(ptr = (adj_vert_t *)graph[i].next;ptr != NULL;ptr = ptr->next)
	    graph[ptr->vertex_num].next_page_rank += value;
	}
      
}

void* parallel_pagerank_update(void* arg){
  int i,j;
  float value;
  struct Thread *thread_data = (struct Thread *)arg;
  adj_vert_t *ptr = NULL;
  double pr_diff;
  pr_diff = 0;

  //printf("%d\n", thread_data->tid);
  for(i = thread_data->start_node;i < thread_data->end_node;i++)
	{
	  pr_diff += fabsf(graph[i].next_page_rank - graph[i].curr_page_rank);
	  graph[i].curr_page_rank = graph[i].next_page_rank;
	  graph[i].next_page_rank = RAND_HOP_LIKELIHOOD(rand_hop,thread_data->num_vertices);
	}
}

int Compute_PageRank(int num_vertices)
{
  long i,j,t;
  //adj_vert_t *ptr = NULL;
  double value = 0;
  double pr_diff;
  pthread_t threads[NUM_THREADS];
  struct Thread Threads_data[NUM_THREADS];
  long iterations=0;
  do 
  {	
      pr_diff = 0;

      for (t=0; t < NUM_THREADS; t++){

	Threads_data[t].tid = t;
	Threads_data[t].start_node = t*(num_vertices/NUM_THREADS);
	Threads_data[t].end_node = ((t+1)*(num_vertices/NUM_THREADS));
	Threads_data[t].num_vertices = num_vertices;
	
	pthread_create(&threads[t], NULL, &parallel_pagerank, (void*) &Threads_data[t]);
      }
      for (t=0; t < NUM_THREADS; t++){
	pthread_join(threads[t], NULL);
      }
     

      for (t=0; t < NUM_THREADS; t++){

	Threads_data[t].tid = t;
	Threads_data[t].start_node = t*(num_vertices/NUM_THREADS);
	Threads_data[t].end_node = ((t+1)*(num_vertices/NUM_THREADS));
	Threads_data[t].num_vertices = num_vertices;
	
	pthread_create(&threads[t], NULL, &parallel_pagerank_update, (void*) &Threads_data[t]);
      }
      for (t=0; t < NUM_THREADS; t++){
	pthread_join(threads[t], NULL);
      }

      printf("\niteration %d done\n",iterations);
      //printf("Diff : %f\n",pr_diff);
      iterations++;
      //     }while(pr_diff > epsilon);
    }while(iterations<14);
  return iterations;
}

int main(int argc, char *argv[])
{
	long i,j,t;
	FILE *file;
	char *token1, *token2;
	char line[MAX_LINE_LEN];
	adj_vert_t *ptr = NULL;
	double value = 0;
	double pr_diff;
	long num_vertices = 0;
	long pnode, cnode;
	long iterations=0;
	struct timespec time_diff;
	struct timespec diff(struct timespec start, struct timespec end);
	struct timespec time1, time2;
	pthread_t threads[NUM_THREADS];
	int rc;
	
	if(argc != 3)
		return -1;
	num_vertices = string_to_long(argv[1]);
	if(num_vertices < 0)
		return -1;
	graph = (vertex_t *)malloc(num_vertices * sizeof(vertex_t));

	epsilon =(double) 0.000001/num_vertices;
	
	

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
			if(append_node(graph,pnode,cnode,num_vertices))
				return -1;
		}
	}
	else
		return -1;
	
	printf("Graph parsing successful\n");

	//struct thread_data thread_data_array[NUM_THREADS];
	//Start time
	clock_gettime(CLOCK_REALTIME, &time1);
	
	iterations=Compute_PageRank(num_vertices);
	
	//End Time
	clock_gettime(CLOCK_REALTIME, &time2);
	time_diff=diff(time1,time2);
	//print_converged_pr_vals(graph,num_vertices);
	printf("Number of iterations: %lu\n",iterations);
	printf("Number of Cycles: %ld\n", (long int)((double)(CPG)*(double)
		 (GIG * time_diff.tv_sec + time_diff.tv_nsec)));
	return 0;
}
