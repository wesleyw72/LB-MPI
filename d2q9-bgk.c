/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <xmmintrin.h>
#include<sys/time.h>
#include<sys/resource.h>
#define SPEED(numspeeds,x,y,numX,numY) ((numspeeds * numY * numX) + (y * numX) + x)
//#define SPEED(numspeeds,x,y,numX,numY) ((y * 9 * numX) + (x*9) + numspeeds)
#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

 int    tot_cells = 0;  /* no. of cells used in calculation */
//float tot_u;          /* accumulated magnitudes of velocity for each cell */
/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;
typedef struct 
{
  int top;
  int bottom;
  int under;
  int above;
  int rankup;
  int rankdown;
  int myrank;
} rankData;
/*
** function prototypes
*/
float collideCollum(const t_param params,float* cells,float* tmp_cells,char* obstacles,int col);
void collate(int size,float* cells,int* rows_per_rank,int* bottom_address,float* av_vels,char* obstacles,const t_param params,int rank,const rankData rankInfo);
/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               char** obstacles_ptr, float** av_vels_ptr,int rank,int size);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
void setup(const t_param params,int rank,int size, rankData* currentRank,int* rows_per_rank,int* bottom_address);
float final_av_velocity(const t_param params, float* cells, char* obstacles);
float timestep(const t_param params, float* cells, float* tmp_cells, char* obstacles,const rankData rankInfo,int size,int tt);
int accelerate_flow(const t_param params, float* cells, char* obstacles);
float collision(const t_param params, float* cells, float* tmp_cells, char* obstacles,const rankData rankInfo,int size);
int write_values(const t_param params, float* cells, char* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             char** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* cells, int rank, int rows_per_rank);

/* compute average velocity */
float av_velocity(const t_param params, float* cells, char* obstacles,int rank,int rows_per_rank);
float calc_reynolds(const t_param params, float* cells, char* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
void printGrid(const t_param params, float* cells,char* name);
/*
** main program:
** initialise, timestep loop, finalise
*/
void swap_pointers(float** cells, float** tmp_cells)
{
  float *temp = *cells;
  *cells = *tmp_cells;
  *tmp_cells = temp;
  //printf("asd\n");
}
void init2(float *cells_ptr, t_param params);
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  // t_speed* cells     = NULL;    /* grid containing fluid densities */
  // t_speed* tmp_cells = NULL;    /* scratch space */
  
  float* cells __attribute__((aligned(32)));
  cells = NULL;
  float* tmp_cells __attribute__((aligned(32)));
  tmp_cells = NULL;
  char*     obstacles __attribute__((aligned(32)));
  rankData rankInfo;
  obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */
  MPI_Init( &argc, &argv );
  int size,rank;
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  int rows_per_rank[size];
  int bottom_address[size];
  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels,rank,size);
  //int rows_per_rank= params.ny/size;
  init2(cells,params);
  setup(params,rank,size,&rankInfo,rows_per_rank,bottom_address);
  
  //printf("Rank %d top: %d bottom %d up %d down %d above %d under %d\n",rank,rankInfo.top,rankInfo.bottom,rankInfo.rankup,rankInfo.rankdown,rankInfo.above,rankInfo.under);
  char tempN[20];
  sprintf(tempN,"PREprintGrid%d",rank);
  //printGrid(params,cells,tempN);
  //printf("tot_cells: %d\n",tot_cells);
  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    //   if(tt==35000)
    // {
    //   char temp[20];
    //   sprintf(temp,"t1printGrid%d",rank);
    //   printGrid(params,cells,temp);
    // }
    
    av_vels[tt] = timestep(params, cells, tmp_cells, obstacles,rankInfo,size,tt);
    
    //av_vels[tt] = av_velocity(params, tmp_cells, obstacles,rank,rows_per_rank);
    
      swap_pointers(&cells,&tmp_cells);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells,rank, rows_per_rank));
#endif
  }
  
  
  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  collate(size,cells,rows_per_rank,bottom_address,av_vels,obstacles,params,rank,rankInfo);
  /* write final values and free memory */
  
  if(rank==0)
  {
    printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    printf("in write vals");
    write_values(params, cells, obstacles, av_vels);
  }
  
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
void collate(int size,float* cells,int* rows_per_rank,int* base_address,float* av_vels,char* obstacles,const t_param params,int rank,const rankData rankInfo)
{
  //printGrid(params,cells,tempN);
  if(rank==0)
  {
    float colAv[params.maxIters];
    //master
    for(int i =1;i<size;i++)
    {
      //printf("I:%d\n",i);
      //gather velocities
      MPI_Recv(colAv,params.maxIters,MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      for(int k = 0;k<params.maxIters;k++)
      {
        av_vels[k]+=colAv[k];
      }
      //Gather grid
      MPI_Recv(&cells[SPEED(0,0,base_address[i],params.nx,params.ny)],params.nx*rows_per_rank[i],MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Recv(&cells[SPEED(8,0,base_address[i],params.nx,params.ny)],params.nx*rows_per_rank[i],MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Recv(&cells[SPEED(2,0,base_address[i],params.nx,params.ny)],params.nx*rows_per_rank[i],MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Recv(&cells[SPEED(7,0,base_address[i],params.nx,params.ny)],params.nx*rows_per_rank[i],MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Recv(&cells[SPEED(5,0,base_address[i],params.nx,params.ny)],params.nx*rows_per_rank[i],MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Recv(&cells[SPEED(3,0,base_address[i],params.nx,params.ny)],params.nx*rows_per_rank[i],MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Recv(&cells[SPEED(1,0,base_address[i],params.nx,params.ny)],params.nx*rows_per_rank[i],MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Recv(&cells[SPEED(4,0,base_address[i],params.nx,params.ny)],params.nx*rows_per_rank[i],MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Recv(&cells[SPEED(6,0,base_address[i],params.nx,params.ny)],params.nx*rows_per_rank[i],MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      //printf("Received Rank %d into address %d\n",i,SPEED(0,0,i*rows_per_rank,params.nx,params.ny));
    }
    //printGrid(params,cells,"gridOut");
  }
  else{
      int numrows = rankInfo.top-rankInfo.bottom+1;
      //printf("Rank: %d numrows:%d\n",rank,numrows);
      MPI_Send(av_vels,params.maxIters,MPI_FLOAT,0,0,MPI_COMM_WORLD);
      MPI_Send(&cells[SPEED(0,0,rankInfo.bottom,params.nx,params.ny)],params.nx*numrows,MPI_FLOAT,0,0,MPI_COMM_WORLD);
      MPI_Send(&cells[SPEED(8,0,rankInfo.bottom,params.nx,params.ny)],params.nx*numrows,MPI_FLOAT,0,0,MPI_COMM_WORLD);
      MPI_Send(&cells[SPEED(2,0,rankInfo.bottom,params.nx,params.ny)],params.nx*numrows,MPI_FLOAT,0,0,MPI_COMM_WORLD);
      MPI_Send(&cells[SPEED(7,0,rankInfo.bottom,params.nx,params.ny)],params.nx*numrows,MPI_FLOAT,0,0,MPI_COMM_WORLD);
      MPI_Send(&cells[SPEED(5,0,rankInfo.bottom,params.nx,params.ny)],params.nx*numrows,MPI_FLOAT,0,0,MPI_COMM_WORLD);
      MPI_Send(&cells[SPEED(3,0,rankInfo.bottom,params.nx,params.ny)],params.nx*numrows,MPI_FLOAT,0,0,MPI_COMM_WORLD);
      MPI_Send(&cells[SPEED(1,0,rankInfo.bottom,params.nx,params.ny)],params.nx*numrows,MPI_FLOAT,0,0,MPI_COMM_WORLD);
      MPI_Send(&cells[SPEED(4,0,rankInfo.bottom,params.nx,params.ny)],params.nx*numrows,MPI_FLOAT,0,0,MPI_COMM_WORLD);
      MPI_Send(&cells[SPEED(6,0,rankInfo.bottom,params.nx,params.ny)],params.nx*numrows,MPI_FLOAT,0,0,MPI_COMM_WORLD);      
  }
}
float timestep(const t_param params, float* cells, float* tmp_cells, char* obstacles,const rankData rankInfo,int size,int tt)
{
  accelerate_flow(params, cells, obstacles);
  return collision(params, cells, tmp_cells, obstacles,rankInfo,size);
  //return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, float* cells, char* obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.0;
  float w2 = params.density * params.accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = params.ny - 2;
  for (int jj = 0; jj < params.nx; jj++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii * params.nx + jj]
        && (cells[SPEED(7,jj,ii,params.nx,params.ny)] - w1) > 0.0
        && (cells[SPEED(1,jj,ii,params.nx,params.ny)] - w2) > 0.0
        && (cells[SPEED(4,jj,ii,params.nx,params.ny)] - w2) > 0.0)
    {
      /* increase 'east-side' densities */
      cells[SPEED(8,jj,ii,params.nx,params.ny)] += w1;
      cells[SPEED(7,jj,ii,params.nx,params.ny)] -= w1;
      cells[SPEED(3,jj,ii,params.nx,params.ny)] += w2;
      cells[SPEED(1,jj,ii,params.nx,params.ny)] -= w2;
      cells[SPEED(4,jj,ii,params.nx,params.ny)] -= w2;
      cells[SPEED(6,jj,ii,params.nx,params.ny)] += w2;
      /* decrease 'west-side' densities */   
    }
  }
  return EXIT_SUCCESS;
}
float Q_sqrt (float number)
{
  if(number==0)
  {
    return 0;
  }
  float k;
  __m128 a = _mm_load_ss(&number);
  __m128 b = _mm_rsqrt_ss(a);
  b = _mm_mul_ss(a,b);
  _mm_store_ss(&k,b);
  return k;
}
void printGrid(const t_param params, float* cells,char* name)
{
  printf("Started printing %s\n",name);
  FILE* fp;
  fp = fopen(name,"w");

  for(int y =0;y<params.ny;y++)
  {
    for(int x=0;x<params.nx;x++)
    {
      for(int s=0;s<9;s++)
      {
        fprintf(fp,"%d %d %d %f\n",x,y,s,cells[SPEED(s,x,y,params.nx,params.ny)]);
      }
    }
  }
  fclose(fp);
  printf("Finished printing %s\n",name);
}
float collideCollum(const t_param params,float* cells,float* tmp_cells,char* obstacles,int col)
{
  float tot_u =0 ;
  int ii= col;
     __assume_aligned(cells,32);
    __assume_aligned(tmp_cells, 32);
    __assume_aligned(obstacles,32);
#pragma simd reduction(+:tot_u)
    for (int jj = 0; jj < params.nx; jj++)
    {
      __assume_aligned(cells,32);
    __assume_aligned(tmp_cells, 32);
    __assume_aligned(obstacles,32);
     int y_n = (ii + 1) % params.ny;
     int x_e = (jj + 1) % params.nx;
     int y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
     int x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);
      //t_speed localSpeeds;
     float t0,t1,t2,t3,t4,t5,t6,t7,t8;
      /* propagate densities to neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      __assume_aligned(cells,32);
      t0 = cells[SPEED(0,jj,ii,params.nx,params.ny)];
      t1 = cells[SPEED(8,x_w,ii,params.nx,params.ny)];
      t2 = cells[SPEED(2,jj,y_s,params.nx,params.ny)];
      t3 = cells[SPEED(7,x_e,ii,params.nx,params.ny)];
      t4 = cells[SPEED(5,jj,y_n,params.nx,params.ny)];
      t5 = cells[SPEED(3,x_w,y_s,params.nx,params.ny)];
      t6 = cells[SPEED(1,x_e,y_s,params.nx,params.ny)];
      t7 = cells[SPEED(4,x_e,y_n,params.nx,params.ny)];
      t8 = cells[SPEED(6,x_w,y_n,params.nx,params.ny)];
        /* compute local density total */
      float local_density = t0;
      local_density += t1;
      local_density += t2;
      local_density += t3;
      local_density += t4;
      local_density += t5;
      local_density += t6;
      local_density += t7;
      local_density += t8;   
        /* compute x velocity component */
      float u_x = (t1+ t5+ t8- (t3+ t6+ t7))/ local_density;
        /* compute y velocity component */
      float u_y = (t2+ t5+ t6- (t4+ t7+ t8))/ local_density;
      const float ld1 =local_density*params.omega/9.0f;
      const float ld2 = local_density*params.omega/36.0f;
      const float u_s = u_x + u_y;
      const float u_d = -u_x +u_y; 
      const float u_sq = (u_x*u_x+u_y*u_y);
      const float d_eq = (1.0f - 1.5f*(u_sq));
      tmp_cells[SPEED(8,jj,ii,params.nx,params.ny)] = obstacles[ii * params.nx + jj] ? t3 : ld1*(d_eq + 4.5f*u_x*(2.0f/3.0f + u_x)) - 0.85f*t1;
      tmp_cells[SPEED(7,jj,ii,params.nx,params.ny)] = obstacles[ii * params.nx + jj] ? t1 :ld1*(d_eq - 4.5f*u_x*(2.0f/3.0f - u_x)) - 0.85f*t3;
      tmp_cells[SPEED(2,jj,ii,params.nx,params.ny)] = obstacles[ii * params.nx + jj] ? t4 :ld1*(d_eq + 4.5f*u_y*(2.0f/3.0f + u_y)) - 0.85f*t2;
      tmp_cells[SPEED(5,jj,ii,params.nx,params.ny)] = obstacles[ii * params.nx + jj] ? t2 :ld1*(d_eq - 4.5f*u_y*(2.0f/3.0f - u_y)) - 0.85f*t4;
      tmp_cells[SPEED(3,jj,ii,params.nx,params.ny)] = obstacles[ii * params.nx + jj] ? t7 :ld2*(d_eq + 4.5f*u_s*(2.0f/3.0f + u_s)) - 0.85f*t5;
      tmp_cells[SPEED(4,jj,ii,params.nx,params.ny)] = obstacles[ii * params.nx + jj] ? t5 :ld2*(d_eq - 4.5f*u_s*(2.0f/3.0f - u_s)) - 0.85f*t7;
      tmp_cells[SPEED(1,jj,ii,params.nx,params.ny)] = obstacles[ii * params.nx + jj] ? t8 :ld2*(d_eq + 4.5f*u_d*(2.0f/3.0f + u_d)) - 0.85f*t6;
      tmp_cells[SPEED(6,jj,ii,params.nx,params.ny)] = obstacles[ii * params.nx + jj] ? t6 :ld2*(d_eq - 4.5f*u_d*(2.0f/3.0f - u_d)) - 0.85f*t8;
      tmp_cells[SPEED(0,jj,ii,params.nx,params.ny)] = d_eq*local_density*params.omega*(4.0f/9.0f) - 0.85f*t0;      
      tot_u += obstacles[ii * params.nx + jj] ? 0 :  sqrt((u_x * u_x) + (u_y * u_y));
    }
    return tot_u;
}
float collision(const t_param params, float* cells, float* tmp_cells, char* obstacles,const rankData rankInfo,int size)
{
  //const float c_sq = 1.0 / 3.0; /* square of speed of sound */
  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  float tot_u;
  MPI_Request outgoing[2];
  MPI_Request incoming[2];
  MPI_Status status;
  MPI_Datatype mygrid;
  MPI_Type_vector(3,params.nx,params.nx*params.ny,MPI_FLOAT,&mygrid);
  MPI_Type_commit(&mygrid);
//send top rows up
//send bottom row down
  int up = rankInfo.rankup;
  int down = rankInfo.rankdown;
  int mytop = rankInfo.top;
  int mybottom = rankInfo.bottom;
  int myunder = rankInfo.under;
  int myabove = rankInfo.above;
  //printf("RANK: %d, UP: %d, DOWN: %d, MYTOP: %d, MYBOT: %d, UNDER:%d, ABOVE: %d\n",rank,up,down,mytop,mybottom,myunder,myabove);
  //upwards
  __assume_aligned(cells,32);
  __assume_aligned(tmp_cells, 32);
  __assume_aligned(obstacles,32);

  MPI_Isend(&cells[SPEED(1,0,mytop,params.nx,params.ny)],1,mygrid,up,0,MPI_COMM_WORLD,&outgoing[0]);
  MPI_Irecv(&cells[SPEED(1,0,myunder,params.nx,params.ny)],1,mygrid,down,0,MPI_COMM_WORLD,&incoming[0]);

  MPI_Isend(&cells[SPEED(4,0,mybottom,params.nx,params.ny)],1,mygrid,down,0,MPI_COMM_WORLD,&outgoing[1]);
  MPI_Irecv(&cells[SPEED(4,0,myabove,params.nx,params.ny)],1,mygrid,up,0,MPI_COMM_WORLD,&incoming[1]);
  
  // MPI_Isend(&cells[SPEED(2,0,mytop,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,MPI_COMM_WORLD,&outgoing[0]);
  // MPI_Irecv(&cells[SPEED(2,0,myunder,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,MPI_COMM_WORLD,&incoming[0]);
  
  // MPI_Isend(&cells[SPEED(1,0,mytop,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,MPI_COMM_WORLD,&outgoing[1]);
  // MPI_Irecv(&cells[SPEED(1,0,myunder,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,MPI_COMM_WORLD,&incoming[1]);
  
  // MPI_Isend(&cells[SPEED(3,0,mytop,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,MPI_COMM_WORLD,&outgoing[2]);
  // MPI_Irecv(&cells[SPEED(3,0,myunder,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,MPI_COMM_WORLD,&incoming[2]);
  
  // MPI_Isend(&cells[SPEED(5,0,mybottom,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,MPI_COMM_WORLD,&outgoing[3]);
  // MPI_Irecv(&cells[SPEED(5,0,myabove,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,MPI_COMM_WORLD,&incoming[3]);
  
  // MPI_Isend(&cells[SPEED(4,0,mybottom,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,MPI_COMM_WORLD,&outgoing[4]);
  // MPI_Irecv(&cells[SPEED(4,0,myabove,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,MPI_COMM_WORLD,&incoming[4]);
  
  // MPI_Isend(&cells[SPEED(6,0,mybottom,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,MPI_COMM_WORLD,&outgoing[5]);
  // MPI_Irecv(&cells[SPEED(6,0,myabove,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,MPI_COMM_WORLD,&incoming[5]);
  //MPI_Sendrecv(&cells[SPEED(2,0,mytop,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,&cells[SPEED(2,0,myunder,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,MPI_COMM_WORLD,&status);
  //MPI_Sendrecv(&cells[SPEED(1,0,mytop,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,&cells[SPEED(1,0,myunder,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,MPI_COMM_WORLD,&status);
  //MPI_Sendrecv(&cells[SPEED(3,0,mytop,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,&cells[SPEED(3,0,myunder,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,MPI_COMM_WORLD,&status);
  // //downwards
  //MPI_Sendrecv(&cells[SPEED(5,0,mybottom,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,&cells[SPEED(5,0,myabove,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,MPI_COMM_WORLD,&status);
  //MPI_Sendrecv(&cells[SPEED(4,0,mybottom,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,&cells[SPEED(4,0,myabove,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,MPI_COMM_WORLD,&status);
  //MPI_Sendrecv(&cells[SPEED(6,0,mybottom,params.nx,params.ny)],params.nx,MPI_FLOAT,down,0,&cells[SPEED(6,0,myabove,params.nx,params.ny)],params.nx,MPI_FLOAT,up,0,MPI_COMM_WORLD,&status);

  for (int ii = mybottom+1; ii < mytop; ii++) 
  {
 tot_u+=collideCollum(params,cells,tmp_cells,obstacles,ii);
  }
  MPI_Waitall(2,incoming,MPI_STATUS_IGNORE);
  tot_u+=collideCollum(params,cells,tmp_cells,obstacles,mybottom);
  tot_u+=collideCollum(params,cells,tmp_cells,obstacles,mytop);
  MPI_Waitall(2,outgoing,MPI_STATUS_IGNORE);
  MPI_Type_free(&mygrid);
  //char tempN[20];
  // sprintf(tempN,"afterSEND%d",rank);
  // printGrid(params,cells,tempN);
  return tot_u/(float)tot_cells;
}
float final_av_velocity(const t_param params, float* cells, char* obstacles)
{
  float tot_u = 0.0;
  /* initialise */
  /* loop over all non-blocked cells */
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii * params.nx + jj])
      {
        /* local density total */
        float local_density = 0.0;

        // for (int kk = 0; kk < NSPEEDS; kk++)
        // {
        //   local_density += cells[ii * params.nx + jj].speeds[kk];
        // }
        
        local_density += cells[SPEED(0,jj,ii,params.nx,params.ny)];
        local_density += cells[SPEED(8,jj,ii,params.nx,params.ny)];
        local_density += cells[SPEED(2,jj,ii,params.nx,params.ny)];
        local_density += cells[SPEED(7,jj,ii,params.nx,params.ny)];
        local_density += cells[SPEED(5,jj,ii,params.nx,params.ny)];
        local_density += cells[SPEED(3,jj,ii,params.nx,params.ny)];
        local_density += cells[SPEED(1,jj,ii,params.nx,params.ny)];
        local_density += cells[SPEED(4,jj,ii,params.nx,params.ny)];
        local_density += cells[SPEED(6,jj,ii,params.nx,params.ny)];


        /* x-component of velocity */
        float u_x = (cells[SPEED(8,jj,ii,params.nx,params.ny)]
                      + cells[SPEED(3,jj,ii,params.nx,params.ny)]
                      + cells[SPEED(6,jj,ii,params.nx,params.ny)]
                      - (cells[SPEED(7,jj,ii,params.nx,params.ny)]
                         + cells[SPEED(1,jj,ii,params.nx,params.ny)]
                         + cells[SPEED(4,jj,ii,params.nx,params.ny)]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[SPEED(2,jj,ii,params.nx,params.ny)]
                      + cells[SPEED(3,jj,ii,params.nx,params.ny)]
                      + cells[SPEED(1,jj,ii,params.nx,params.ny)]
                      - (cells[SPEED(5,jj,ii,params.nx,params.ny)]
                         + cells[SPEED(4,jj,ii,params.nx,params.ny)]
                         + cells[SPEED(6,jj,ii,params.nx,params.ny)]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        //++tot_cells;
      }
      
    }
  }
  
  return tot_u / (float)tot_cells;
}
float av_velocity(const t_param params, float* cells, char* obstacles,int rank,int rows_per_rank)
{
  float tot_u = 0.0;
  /* initialise */
  int mytop = rank*rows_per_rank+rows_per_rank-1;
  int mybottom = rank*rows_per_rank;
  /* loop over all non-blocked cells */
  for (int ii = mybottom; ii <= mytop; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii * params.nx + jj])
      {
        /* local density total */
        int y_n = (ii + 1) % params.ny;
     int x_e = (jj + 1) % params.nx;
     int y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
     int x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);
      //t_speed localSpeeds;
     float t0,t1,t2,t3,t4,t5,t6,t7,t8;
      /* propagate densities to neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      __assume_aligned(cells,32);
      t0 = cells[SPEED(0,jj,ii,params.nx,params.ny)];
      t1 = cells[SPEED(8,x_w,ii,params.nx,params.ny)];
      t2 = cells[SPEED(2,jj,y_s,params.nx,params.ny)];
      t3 = cells[SPEED(7,x_e,ii,params.nx,params.ny)];
      t4 = cells[SPEED(5,jj,y_n,params.nx,params.ny)];
      t5 = cells[SPEED(3,x_w,y_s,params.nx,params.ny)];
      t6 = cells[SPEED(1,x_e,y_s,params.nx,params.ny)];
      t7 = cells[SPEED(4,x_e,y_n,params.nx,params.ny)];
      t8 = cells[SPEED(6,x_w,y_n,params.nx,params.ny)];
        /* compute local density total */
      float local_density = t0;
      local_density += t1;
      local_density += t2;
      local_density += t3;
      local_density += t4;
      local_density += t5;
      local_density += t6;
      local_density += t7;
      local_density += t8;   
        /* x-component of velocity */
        float u_x = (cells[SPEED(8,jj,ii,params.nx,params.ny)]
                      + cells[SPEED(3,jj,ii,params.nx,params.ny)]
                      + cells[SPEED(6,jj,ii,params.nx,params.ny)]
                      - (cells[SPEED(7,jj,ii,params.nx,params.ny)]
                         + cells[SPEED(1,jj,ii,params.nx,params.ny)]
                         + cells[SPEED(4,jj,ii,params.nx,params.ny)]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[SPEED(2,jj,ii,params.nx,params.ny)]
                      + cells[SPEED(3,jj,ii,params.nx,params.ny)]
                      + cells[SPEED(1,jj,ii,params.nx,params.ny)]
                      - (cells[SPEED(5,jj,ii,params.nx,params.ny)]
                         + cells[SPEED(4,jj,ii,params.nx,params.ny)]
                         + cells[SPEED(6,jj,ii,params.nx,params.ny)]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        //++tot_cells;
      } 
    }
  }
  return tot_u / (float)tot_cells;
}
void init2(float *cells_ptr, t_param params)
{
  float w0 = params.density * 4.0 / 9.0;
  float w1 = params.density      / 9.0;
  float w2 = params.density      / 36.0;
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* centre */
      cells_ptr[SPEED(0,jj,ii,params.nx,params.ny)] = w0;
      /* axis directions */
      cells_ptr[SPEED(8,jj,ii,params.nx,params.ny)] = w1;
      cells_ptr[SPEED(2,jj,ii,params.nx,params.ny)] = w1;
      cells_ptr[SPEED(7,jj,ii,params.nx,params.ny)] = w1;
      cells_ptr[SPEED(5,jj,ii,params.nx,params.ny)] = w1;
      /* diagonals */
      cells_ptr[SPEED(3,jj,ii,params.nx,params.ny)] = w2;
      cells_ptr[SPEED(1,jj,ii,params.nx,params.ny)] = w2;
      cells_ptr[SPEED(4,jj,ii,params.nx,params.ny)] = w2;
      cells_ptr[SPEED(6,jj,ii,params.nx,params.ny)] = w2;
    }
  }
}
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               char** obstacles_ptr, float** av_vels_ptr,int rank,int size)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);
  //setup(*params,rank,size);
  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (float*)_mm_malloc(sizeof(t_speed) * (params->ny * params->nx),32);

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (float*)_mm_malloc(sizeof(t_speed) * (params->ny * params->nx),32);

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(char) * (params->ny * params->nx),32);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
 //s
  /* initialise densities */
  // float w0 = params->density * 4.0 / 9.0;
  // float w1 = params->density      / 9.0;
  // float w2 = params->density      / 36.0;
  // #pragma omp parallel for schedule(static)
  // for (int ii = 0; ii < params->ny; ii++)
  // {
  //   for (int jj = 0; jj < params->nx; jj++)
  //   {
  //     /* centre */
  //     (*cells_ptr)[SPEED(0,jj,ii,params.nx,params.ny)] = w0;
  //     /* axis directions */
  //     (*cells_ptr)[SPEED(8,jj,ii,params.nx,params.ny)] = w1;
  //     (*cells_ptr)[SPEED(2,jj,ii,params.nx,params.ny)] = w1;
  //     (*cells_ptr)[SPEED(7,jj,ii,params.nx,params.ny)] = w1;
  //     (*cells_ptr)[SPEED(5,jj,ii,params.nx,params.ny)] = w1;
  //     /* diagonals */
  //     (*cells_ptr)[SPEED(3,jj,ii,params.nx,params.ny)] = w2;
  //     (*cells_ptr)[SPEED(1,jj,ii,params.nx,params.ny)] = w2;
  //     (*cells_ptr)[SPEED(4,jj,ii,params.nx,params.ny)] = w2;
  //     (*cells_ptr)[SPEED(6,jj,ii,params.nx,params.ny)] = w2;
  //   }
  // }
  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
    ++tot_cells;
    /* assign to array */
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
  }
  
  tot_cells = params->nx * params->ny - tot_cells;
  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             char** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  _mm_free(*cells_ptr);
  *cells_ptr = NULL;

  _mm_free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float* cells, char* obstacles)
{
  const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  return final_av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells, int rank, int rows_per_rank)
{
  float total = 0.0;  /* accumulator */
  int mytop = rank*rows_per_rank+rows_per_rank-1;
  int mybottom = rank*rows_per_rank;
  for (int ii = mybottom; ii <= mytop; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[SPEED(kk,jj,ii,params.nx,params.ny)];
      }
    }
  }

  return total;
}

int write_values(const t_param params, float* cells, char* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.0 / 3.0; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* an occupied cell */
      if (obstacles[ii * params.nx + jj])
      {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[SPEED(kk,jj,ii,params.nx,params.ny)];
        }

        /* compute x velocity component */
        u_x = (cells[SPEED(8,jj,ii,params.nx,params.ny)]
               + cells[SPEED(3,jj,ii,params.nx,params.ny)]
               + cells[SPEED(6,jj,ii,params.nx,params.ny)]
               - (cells[SPEED(7,jj,ii,params.nx,params.ny)]
                  + cells[SPEED(1,jj,ii,params.nx,params.ny)]
                  + cells[SPEED(4,jj,ii,params.nx,params.ny)]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[SPEED(2,jj,ii,params.nx,params.ny)]
               + cells[SPEED(3,jj,ii,params.nx,params.ny)]
               + cells[SPEED(1,jj,ii,params.nx,params.ny)]
               - (cells[SPEED(5,jj,ii,params.nx,params.ny)]
                  + cells[SPEED(4,jj,ii,params.nx,params.ny)]
                  + cells[SPEED(6,jj,ii,params.nx,params.ny)]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
void setup(const t_param params,int rank,int size,rankData* currentRank,int* rows_per_rank,int* bottom_address)
{
  if(rank==0)
  {
    //distribute work
    int base_rows_per_rank = params.ny/size;
    //int rows_per_rank[size];
    //int bottom_address[size];
    for(int i =0;i<size;i++)
    {
      rows_per_rank[i]=base_rows_per_rank;
    }
    int remain = params.ny - base_rows_per_rank*size;
    int count = 0;
    while(remain>0)
    {
      rows_per_rank[count]++;
      remain--;
      count++;
      count = count %size;
    }
    int accumulator = 0;
    for(int i =0;i<size;i++)
    {
      bottom_address[i] = accumulator;
      accumulator+=rows_per_rank[i];
    }
    //TODO:Setup Rank 0
    for(int i = 1;i<size;i++)
    {
      //send each num of ranks out followed by rank info
      MPI_Send(&rows_per_rank[i],1,MPI_INT,i,0,MPI_COMM_WORLD); //send number of rows in rank
      MPI_Send(&bottom_address[i],1,MPI_INT,i,0,MPI_COMM_WORLD); //Send bottom address
      //printf("RANK: %d rowS: %d \n",i,rows_per_rank[i]);
    }
    int numrows = rows_per_rank[0];
    int bottom = bottom_address[0];
    currentRank->rankup = (rank+1)%size;
    currentRank->rankdown = (rank==0) ? size-1 : rank-1;
    currentRank->bottom = bottom;
    currentRank->top = bottom+numrows-1;
    currentRank->under = (bottom==0) ? params.ny-1 : bottom-1;
    currentRank->above=(currentRank->top+1)%params.ny;
  }
  else
  {
    int numrows;
    int bottom;
    MPI_Recv(&numrows,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(&bottom,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    currentRank->rankup = (rank+1)%size;
    currentRank->rankdown = (rank==0) ? size-1 : rank-1;
    currentRank->bottom = bottom;
    currentRank->top = bottom+numrows-1;
    currentRank->under = (bottom==0) ? params.ny-1 : bottom-1;
    currentRank->above=(currentRank->top+1)%params.ny;

    //receive work
  }

}
