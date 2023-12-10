#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "mpi.h"

#define N 2048
#define gen 5

int get_neighbors(float** grid, int i, int j) {
    int x_row[] = {-1, -1, -1,  0, 0,  1, 1, 1};
    int y_column[] = {-1,  0,  1, -1, 1, -1, 0, 1};
    int quant = 0;

    for (int k = 0; k < 8; k++) {
        int x = (i + x_row[k] + N) % N; 
        int y = (j + y_column[k] + N) % N; 

        if (grid[x][y] > 0) {
            quant += 1;
        }
    }
    return quant;
}

float mean_from_neighbors(float **grid, int i, int j){
    int x_row[] = {-1, -1, -1,  0, 0,  1, 1, 1};
    int y_column[] = {-1,  0,  1, -1, 1, -1, 0, 1};
    float soma = 0;

    for (int k = 0; k < 8; k++) {
        int x = (i + x_row[k] + N) % N; 
        int y = (j + y_column[k] + N) % N; 

        soma += grid[x][y];
    }
    return soma/8;
}


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
  int global_sum = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  float **grid = alloc_grid();
  float **new_grid = alloc_grid();
  float **swap;

  int lin = 1, col = 1;
  int quant_lines_per_process = N/world_size;
  int alives = 0;

  if(world_rank == 0){

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

  double start_time = MPI_Wtime();
  
  for (int i = 0; i < 2000; i++) {
    alives = 0;
    // printf("word_rank = %d, anterior=%d, prox=%d, N=%d, LINHA_ENVIADA_TOP=%d, LINHA_RECEBIDA_TOP=%d LINHA_ENVIADA_BOT=%d LINHA_RECEBI_BOT=%d\n", world_rank, top_neighbor, bottom_neighbor, quant_lines_per_process, world_rank*quant_lines_per_process, (top_neighbor+1)*quant_lines_per_process - 1, (world_rank+1) * quant_lines_per_process-1, bottom_neighbor * quant_lines_per_process);

    memcpy(send_buffer_top, grid[world_rank * quant_lines_per_process], N * sizeof(float));

    MPI_Sendrecv(send_buffer_top, N, MPI_FLOAT, top_neighbor, 0, 
                recv_buffer_bottom, N, MPI_FLOAT, bottom_neighbor, 0, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    memcpy(grid[bottom_neighbor * quant_lines_per_process], recv_buffer_bottom, N * sizeof(float));

    memcpy(send_buffer_bottom, grid[(world_rank+1) * quant_lines_per_process-1], N * sizeof(float));

    MPI_Sendrecv(send_buffer_bottom, N, MPI_FLOAT, bottom_neighbor, 0, 
                recv_buffer_top, N, MPI_FLOAT, top_neighbor, 0, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    memcpy(grid[(top_neighbor + 1)* quant_lines_per_process-1], recv_buffer_top, N * sizeof(float));
  
    //  printf("rank = %d, inicio= %d, fim = %d\n", world_rank,world_rank*quant_lines_per_process, (world_rank+1)*quant_lines_per_process);
    
    for (int i = world_rank*quant_lines_per_process; i < (world_rank+1)*quant_lines_per_process; i++) {
        for (int j = 0; j < N; j++) {
            
            switch(get_neighbors(grid, i, j)){
                case 2:
                    if(grid[i][j] > 0){
                        new_grid[i][j] =  mean_from_neighbors(grid, i, j) > 0 ? 1.0 : 0;
                        alives++;
                    }else new_grid[i][j] = 0;
                    break;
                case 3:
                    new_grid[i][j] =  mean_from_neighbors(grid, i, j) > 0 ? 1.0 : 0;
                    alives++;
                    break;
                default:
                    new_grid[i][j] = 0;
                    break;
                
            }
        }
    }

    swap = grid;
    grid = new_grid;
    new_grid = swap;
    
    // printf("world rank = %d %d alives\n", world_rank, alives);

    MPI_Reduce(&alives, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
  }
  double end_time = MPI_Wtime();
  double elapsed_time = end_time - start_time;
  if(world_rank == 0){
    printf("global_sum = %d alives\n", global_sum);
    printf("Tempo decorrido: %f segundos\n", elapsed_time);

  }
  
  desalloc_grid(grid);
  desalloc_grid(new_grid);
  MPI_Finalize();
}