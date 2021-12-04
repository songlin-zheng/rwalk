#include "rwalk.cuh"
#include <stdio.h>
#include <assert.h>
#include <limits>

int64_t* dev_global_walk;
int64_t* dev_node_idx;
float* dev_timestamp;
int64_t* dev_start_idx;

#define cudaCheck(err) { \
	if (err != cudaSuccess) { \
		printf("CUDA error: %s: %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		assert(err == cudaSuccess); \
	} \
}


// rand_walk -> [num_of_node, num_of_walk, max_walk_length]
void __global__ singleRandomWalk(int num_of_node, int num_of_walk, int max_walk_length, int64_t* node_idx, float* timestamp, int64_t* start_idx, int64_t* rand_walk, unsigned long long rnumber){
    // assuming grid = 1
    int64_t i =  (blockDim.x * blockIdx.x) + threadIdx.x;
    if(i >= num_of_node * num_of_walk){
        return;
    }

    int64_t src_node = i / num_of_walk;
    float curr_timestamp = .0f;
    rand_walk[i * max_walk_length + 0] = src_node;

    int walk_cnt;
    for(walk_cnt = 1; walk_cnt < max_walk_length; walk_cnt ++){
        int64_t start = start_idx[src_node];
        int64_t end = start_idx[src_node];

        // control divergence
        // range should be [start, end)
        if(start < end){
            float* cdf = (float*) malloc((end - start) * sizeof(float));
            int64_t* valid_node_idx = (int64_t*) malloc((end - start) * sizeof(int64_t));
            int idx = 0;
            int valid_neighbor_cnt = 0;
            // float cdf[end - start];
            float max_timestamp = timestamp[start];
            float min_timestamp = timestamp[start];
            // ! parallizable
            for(int j = start; j < end; j ++){
                if(timestamp[j] > curr_timestamp){
                    valid_neighbor_cnt ++;
                    valid_node_idx[idx] = j;
                    cdf[idx++] = timestamp[j];
                }
                max_timestamp = max(curr_timestamp, timestamp[j]);
                min_timestamp = min(min_timestamp, timestamp[j]);
            }
            if(!valid_neighbor_cnt){
                free(valid_node_idx);
                free(cdf);
                break;
            }

            // every timestamp is the same
            if(max_timestamp - min_timestamp >= - 0.0000001 && max_timestamp - min_timestamp <= 0.0000001){
                rand_walk[i * max_walk_length + walk_cnt] = valid_node_idx[0];
                src_node = valid_node_idx[0];
                curr_timestamp = cdf[0];
                free(valid_node_idx);
                free(cdf);
                continue;
            }

            // ! need to determine how to get prob
            float prob = rnumber * 1.0 / ULLONG_MAX;

            // refresh rnumber
            rnumber = rnumber * (unsigned long long)25214903917 + 11;
            bool fall_through = true;

            // ! reduction tree here (kernel in kernel)
            float denom = .0f;
            for(int j = 0; j < valid_neighbor_cnt; j ++){
                cdf[j] =  expf((cdf[j] - curr_timestamp) / (max_timestamp - min_timestamp));
                denom += cdf[j];
            }
            float curr_cdf = 0,  next_cdf = .0f;
            for(int j = 0; j < valid_neighbor_cnt; j ++){
                next_cdf += cdf[j] / denom;
                if(prob >= curr_cdf && prob < next_cdf){
                    src_node = node_idx[start + j];
                    curr_timestamp = timestamp[start + j];
                    fall_through = false;
                    break;
                }
                curr_cdf = next_cdf;
            }

            // fall through should never happen
            if(fall_through){
                rand_walk[i * max_walk_length + walk_cnt] = valid_node_idx[0];
                src_node = valid_node_idx[0];
                curr_timestamp = cdf[0];
            }

            free(valid_node_idx);
            free(cdf);
        }
    }

    if(walk_cnt < max_walk_length){
        // signal the rest is invalid and there is no descending node
        rand_walk[i * max_walk_length + walk_cnt] = -1;
    }
}




void cuda_rwalk(int max_walk_length, int num_walks_per_node, int64_t num_nodes, int64_t num_edges, unsigned long long random_number){

    // malloc GPU memory
    cudaCheck(cudaMalloc((void **)&dev_start_idx, sizeof(int64_t) * (num_nodes + 1)));
    cudaCheck(cudaMalloc((void **)&dev_node_idx, sizeof(int64_t) * num_edges));
    cudaCheck(cudaMalloc((void **)&dev_timestamp, sizeof(float) * num_edges));
    cudaCheck(cudaMalloc((void **)&dev_global_walk, sizeof(int64_t) * num_nodes * max_walk_length * num_walks_per_node));

    // memcpy
    cudaCheck(cudaMemcpy(dev_start_idx, start_idx_host, sizeof(int64_t) * (num_nodes + 1), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_node_idx, node_idx_host, sizeof(int64_t) * num_edges, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_timestamp, timestamp_host, sizeof(float) * num_edges, cudaMemcpyHostToDevice));

    // start training
    uint grid_size = (num_nodes - 1) / 128 + 1;
    dim3 gridDim(grid_size);
    dim3 blockDim(128);
    // ?? header file
    singleRandomWalk<<<gridDim, blockDim>>>(num_nodes, num_walks_per_node, max_walk_length, dev_node_idx, dev_timestamp, dev_start_idx, dev_global_walk, random_number);

    // get result
    cudaCheck(cudaMemcpy(random_walk_host, dev_global_walk, sizeof(int64_t) * num_nodes * max_walk_length * num_walks_per_node, cudaMemcpyDeviceToHost));

    // clean arrays
    cudaCheck(cudaFree(dev_start_idx));
    cudaCheck(cudaFree(dev_node_idx));
    cudaCheck(cudaFree(dev_timestamp));
    cudaCheck(cudaFree(dev_global_walk));
}

