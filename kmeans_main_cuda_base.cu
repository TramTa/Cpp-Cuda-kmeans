#include <iostream>
#include <fstream>
#include <limits>   
#include <stdio.h>       
#include <math.h>   
#include <chrono>
#include <cuda_runtime_api.h>

#include "argparse.h"
#include "kmeans_cuda_basic.h"

using namespace std;

// ===================== 
double distance_cpu(double *x, double *y, int dim)
{
    // start at ptr x and ptr y, and compute distance for d numbers 
    double res = 0; 

    for(int i=0; i<dim; i++){
        res += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sqrt(res);
}


// =============================== 
static unsigned long int next2 = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next2 = next2 * 1103515245 + 12345;
    return (unsigned int)(next2/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next2 = seed;
}


// =============================== MAIN 
int main(int argc, char **argv)
{
    int dim, n_cluster, n_point, max_iter ;
    double err_threshold;

    double double_inf = std::numeric_limits<double>::infinity();  

    struct options_t opts;   

    // read in cmd line options:  "k:d:i:m:t:cs:" 
    get_opts(argc, argv, &opts);

    dim = opts.dim; 
    n_cluster = opts.n_cluster;
    err_threshold = opts.err_threshold;
    max_iter = opts.max_iter;

    // input file 
    std::ifstream in;
    in.open(opts.input_file);  

    in >> n_point;  

    // host data 
    double *h_points        = (double*) malloc( n_point *dim *sizeof(double) );
    int *h_points_assign    = (int*) malloc( n_point *sizeof(int) );

    double *h_clusters      = (double*) malloc( n_cluster *dim *sizeof(double) );
    double *h_clusters_new  = (double*) malloc( n_cluster *dim *sizeof(double) );  
    int *h_clusters_size    = (int*) malloc( n_cluster *sizeof(int) );   

    // set up points and points_assign
    double tmp;
    for(int i=0; i < n_point; i++){
        h_points_assign[i] = 0;   

        in >> tmp;  // point id 

        for(int d=0; d < dim; d++){
            in >> h_points[i*dim + d];
        }
    }


    // initialize clusters/centroids , clusters_new, clusters_size 
    kmeans_srand(opts.seed);  // kmeans_srand( 8675309 );  
    
    for (int i=0; i < n_cluster; i++)
    {
        h_clusters_size[i] = 0; 

        int index = kmeans_rand() % n_point;
        
        for(int d=0; d < dim; d++){
            h_clusters[ i*dim +d ] = h_points[ index*dim +d ];  
            h_clusters_new[ i*dim +d ] = 0;
        }
    }
 
    // set up timer, total time is total time of all iteration
    // data_transfer_time is to transfer data between host and device 
    float memsettime = 0, run_time = 0, data_transfer_time = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // time for data transfer from host to dev 
    cudaEventRecord(start, 0);

    // device data 
    double  *d_points,          *d_clusters,    	*d_clusters_new;
    int     *d_points_assign,   *d_clusters_size; 

    cudaMalloc( &d_points,          n_point *dim *sizeof(double) ); //  n_point *dim
    cudaMalloc( &d_points_assign,   n_point *sizeof(int) );         //  n_point 
    
    cudaMalloc( &d_clusters,        n_cluster *dim *sizeof(double) );   //  len n_cluster *dim
    cudaMalloc( &d_clusters_new,    n_cluster *dim *sizeof(double) );   //  len n_cluster *dim
    cudaMalloc( &d_clusters_size,   n_cluster *sizeof(int) );           //  len n_cluster 

    cudaMemcpy( d_points, h_points, n_point *dim *sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( d_points_assign, h_points_assign, n_point *sizeof(int), cudaMemcpyHostToDevice );
   
    cudaMemcpy( d_clusters, h_clusters, n_cluster *dim *sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( d_clusters_new, h_clusters_new, n_cluster *dim *sizeof(double), cudaMemcpyHostToDevice );

    cudaMemcpy( d_clusters_size, h_clusters_size, n_cluster *sizeof(int), cudaMemcpyHostToDevice ); 

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);       
    cudaEventElapsedTime(&memsettime, start, stop);
    data_transfer_time += memsettime;

    // flag to check convergence 
    int *h_is_more_than_threshold = (int*)malloc( 1*sizeof(int) );
    *h_is_more_than_threshold = 0;

    int *d_is_more_than_threshold;
    cudaMalloc( &d_is_more_than_threshold, sizeof(int) );
    cudaMemcpy( d_is_more_than_threshold, h_is_more_than_threshold, sizeof(int), cudaMemcpyHostToDevice );
        

    int blk_dim = 1024; // ; //  512; 128;
    int iter = 0;

    // ======================
    while(1)
    {
        cudaEventRecord(start, 0);
      	cudaMemset( d_is_more_than_threshold, 0.0, 1*sizeof(int) ); // 0 means converge

        assign_centroid<<< (n_point*dim -1)/blk_dim +1, blk_dim >>>(d_points, d_points_assign, d_clusters, d_clusters_new, d_clusters_size, 
        	n_point, n_cluster, dim, err_threshold, max_iter, double_inf 
        );  

        // cudaDeviceSynchronize();

        update_centroid<<< (n_point*dim -1)/blk_dim +1, blk_dim >>>(d_points, d_points_assign, d_clusters, d_clusters_new, d_clusters_size, 
        	n_point, n_cluster, dim, err_threshold, max_iter, 
            d_is_more_than_threshold 
        );  

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);       
        cudaEventElapsedTime(&memsettime, start, stop);
        run_time += memsettime;

        cudaMemcpy(h_is_more_than_threshold, d_is_more_than_threshold, sizeof(int), cudaMemcpyDeviceToHost);

        iter++;

        if(iter >= max_iter || *h_is_more_than_threshold ==0)
            break;
    } // end of while


    // record time for data transfer from dev to host 
    cudaEventRecord(start, 0);

    cudaMemcpy( h_clusters, d_clusters, n_cluster *dim *sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( h_points_assign, d_points_assign, n_point *sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);       
    cudaEventElapsedTime(&memsettime, start, stop);
    data_transfer_time += memsettime;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // if -c is specified, print centroid of all clusters 
    if(opts.print_centroid)
    {
        for(int i=0; i < n_cluster; i++){            
            printf("%d ", i); // cluster id.

            for(int d=0; d < dim; d++){
                printf( "%lf ", h_clusters[i*dim + d] );
            }
            printf("\n");
        }
    }

    // if -c is not specified, print points labels
    else {
        printf("clusters:");
        for (int p=0; p < n_point; p++){
            printf(" %d", h_points_assign[p]);             
        }
        printf("\n"); 
    }

 
// std::cout 
// << "\n=== CUDA-base ==="
// << "\ndata       			: " << opts.input_file 
// << "\ntotal time  			: " << run_time + data_transfer_time << " ms"
// << "\nnum iter  			: " << iter 
// << "\nrun time  			: " << run_time << " ms"
// << "\nrun time per iteration	: " << run_time/iter << " ms"

// << "\ndata_transfer_time 	: " << data_transfer_time << " ms" 
// << "\ndata_transfer_time / total time 	: " << (data_transfer_time *100)/(data_transfer_time + run_time) << " %" 

// << "\n===" << std::endl;

// =========================================   

    free(h_points);
    free(h_points_assign);
    free(h_clusters);
    free(h_clusters_new);
    free(h_clusters_size);

    cudaFree(d_points);
    cudaFree(d_points_assign);
    cudaFree(d_clusters);
    cudaFree(d_clusters_new);
    cudaFree(d_clusters_size);

    return 0;
}


