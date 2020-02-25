#ifndef CUDA_SEARCH_CUH
#define CUDA_SEARCH_CUH

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include "search.hpp"

#define CUDA_CHECK_RETURN(value) { gpuAssert((value), __FILE__, __LINE__); }
/**
 * Check for errors in return values of CUDA functions
 */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template <typename T>
class CudaSearch : public Search<T> {
public:
    CudaSearch(const std::vector< std::vector<T> > &dataset, int blockSize);
    int getSpaceDim() override;
    void search(T* host_query, std::vector<int> &nnIndexes, std::vector<T> &nnDistancesSqr, const int &numResults) override;
    virtual ~CudaSearch();
private:
    T* dataset;
    T* query;
    int* nnIndexes;
    T* nnDistancesSqr;
    int datasetSize;
    int spaceDim;
    int blockSize;
};


template<typename T>
int CudaSearch<T>::getSpaceDim() {
    return spaceDim;
}

template <typename T>
CudaSearch<T>::CudaSearch(const std::vector< std::vector<T> > &dataset, int blockSize) : datasetSize(dataset.size()), blockSize(blockSize){

    assert(datasetSize > 0);
    this->spaceDim = dataset[0].size();

    // memory allocation on device
    CUDA_CHECK_RETURN(
            cudaMalloc((void ** )&this->dataset, sizeof(T) * datasetSize * spaceDim)
    );
    CUDA_CHECK_RETURN(
            // TODO se si permette le query a batch, questo va cambiato
            cudaMalloc((void ** )&this->query, sizeof(T) * spaceDim)
    );
    CUDA_CHECK_RETURN(
            cudaMalloc((void ** )&this->nnIndexes, sizeof(int) * datasetSize)
    );
    CUDA_CHECK_RETURN(
            cudaMalloc((void ** )&this->nnDistancesSqr, sizeof(T) * datasetSize)
    );

    // dataset copy to device
    for(int i=0; i < datasetSize; i++){
        CUDA_CHECK_RETURN(
                cudaMemcpy(this->dataset + (i*spaceDim), &dataset[i][0], sizeof(T) * spaceDim, cudaMemcpyHostToDevice)
        );
    }

}

template <typename T>//, typename BLOCK_SIZE>
__global__ void _cudaDistances(const T *__restrict__ dataset, const T *__restrict__ query,
                               int *__restrict__ nnIndexes, T *__restrict__ nnDistancesSqr, const int datasetSize,
                               const int spaceDim, int blockSize){
    int i;
    i = blockIdx.x * blockSize + threadIdx.x;
    if(i >= datasetSize)
        return;
    T dist = 0;
    int j;
    const T* dataset_tmp = dataset + (i*spaceDim);
    for(j=0; j < spaceDim; j++){
        const T diff = query[j] - *(dataset_tmp++);
        dist = dist + (diff * diff);
    }

    nnDistancesSqr[i] = dist;
    nnIndexes[i] = i;
}

template<typename T>
void CudaSearch<T>::
search(T* host_query, std::vector<int> &host_nnIndexes, std::vector<T> &host_nnDistancesSqr, const int &numResults){

    // TODO implementare batch search (forse meglio fare un altro metodo)

    // copy query to device memory
    CUDA_CHECK_RETURN(
            // TODO capire per quale motivo questo trasferimento è più lento se la query si trova nella memoria pinned (forse è troppo piccola, e usando il batch le cose cambiano)
            cudaMemcpy(this->query, host_query, sizeof(T) * spaceDim, cudaMemcpyHostToDevice)
    );

    // calculate distances between query and each dataset point
    int numBlocks = (datasetSize + blockSize - 1) / blockSize;
    _cudaDistances <<<numBlocks,blockSize>>>(this->dataset, this->query, this->nnIndexes, this->nnDistancesSqr, this->datasetSize, spaceDim, blockSize);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // sort by increasing distance
    thrust::device_ptr<T> device_thrustDistances(this->nnDistancesSqr);
    thrust::device_ptr<int> device_thrustIndexes(this->nnIndexes);
    thrust::sort_by_key(device_thrustDistances, device_thrustDistances+this->datasetSize, device_thrustIndexes);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // copy results from device to host memory
    CUDA_CHECK_RETURN(
            cudaMemcpy(&host_nnIndexes[0], this->nnIndexes, sizeof(int) * numResults, cudaMemcpyDeviceToHost)
    );
    CUDA_CHECK_RETURN(
            cudaMemcpy(&host_nnDistancesSqr[0], this->nnDistancesSqr, sizeof(T) * numResults, cudaMemcpyDeviceToHost)
    );
}


template <typename T>
CudaSearch<T>::~CudaSearch() {
    // free device memory
    cudaFree(this->dataset);
    cudaFree(this->nnIndexes);
    cudaFree(this->nnDistancesSqr);
    cudaFree(this->query);
}

#endif //CUDA_SEARCH_CUH
