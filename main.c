#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "mpi.h"

#define N 32
#define gen 5

float** alloc_grid(){
  float **grid  = (float**) calloc(sizeof(float*),N);
  for(int i = 0; i < N; i++){
    grid[i] =  (float*) calloc(sizeof(float), N);
  }
  return grid;
}

void desalloc_grid(float** grid){
  for(int i = 0; i < N; i++){
    free(grid[i]);
  }

  free(grid);
}


int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  float **grid = alloc_grid();
  float **new_grid = alloc_grid();

  int lin = 1, col = 1;
  int quant_lines_per_process = N/world_size;

  if(world_rank == 0){
    grid[0 ][0] = 11111.0;

    grid[lin ][col+1] = 1.0;
    grid[lin+1][col+2] = 1.0;
    grid[lin+2][col  ] = 1.0;
    grid[lin+2][col+1] = 1.0;
    grid[lin+2][col+2] = 1.0;


    lin =10; col = 30;
    grid[lin  ][col+1] = 1.0;
    grid[lin  ][col+2] = 1.0;
    grid[lin+1][col  ] = 1.0;
    grid[lin+1][col+1] = 1.0;
    grid[lin+2][col+1] = 1.0;

  }

  int top_neighbor = (world_rank == 0) ? world_size - 1 : world_rank - 1;
  int bottom_neighbor = (world_rank == world_size - 1) ? 0 : world_rank + 1;

  float* send_buffer_top = (float*)malloc(N * sizeof(float));
  float* recv_buffer_top = (float*)malloc(N * sizeof(float));

  float* send_buffer_bottom = (float*)malloc(N * sizeof(float));
  float* recv_buffer_bottom = (float*)malloc(N * sizeof(float));
  
  for (int i = 0; i < 1; i++) {

    printf("word_rank = %d, anterior=%d, prox=%d, N=%d, LINHA_ENVIADA=%d, LINHA_RECEBIDA=%d\n", world_rank, top_neighbor, bottom_neighbor, quant_lines_per_process, world_rank*quant_lines_per_process, (top_neighbor+1)*quant_lines_per_process - 1);

    memcpy(send_buffer_top, grid[world_rank * quant_lines_per_process], N * sizeof(float));

    MPI_Sendrecv(send_buffer_top, N, MPI_FLOAT, top_neighbor, 0, 
                recv_buffer_bottom, N, MPI_FLOAT, bottom_neighbor, 0, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    memcpy(new_grid[(top_neighbor + 1) * quant_lines_per_process - 1], recv_buffer_bottom, N * sizeof(float));

    memcpy(send_buffer_bottom, grid[ ((world_rank+1) * quant_lines_per_process) - 1], N * sizeof(float));

    MPI_Sendrecv(send_buffer_bottom, N, MPI_FLOAT, bottom_neighbor, 0, 
                recv_buffer_top, N, MPI_FLOAT, top_neighbor, 0, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    memcpy(new_grid[(bottom_neighbor + 1) * quant_lines_per_process - 1], recv_buffer_top, N * sizeof(float));
  
    for(int i = 0; i < N; i++){
      for(int j = 0; j < N; j++){
        if(new_grid[i][j] != 0)
          printf("world rank = %d, %f %d %d\n", world_rank, new_grid[i][j], i, j);
      }
      //printf("\n");
    }

    
    
    MPI_Barrier(MPI_COMM_WORLD);
  }

  desalloc_grid(grid);
  desalloc_grid(new_grid);
  MPI_Finalize();
}