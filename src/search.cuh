#ifndef CUDA_SEARCH_CUH
#define CUDA_SEARCH_CUH

#include <cuda.h>
#include <iostream>
#include <vector>
#include <thrust/sort.h>
#include "search.hpp"
#define BLOCK_SIZE 1024

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
    CudaSearch(const std::vector< std::vector<T> > &dataset);
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
};


template<typename T>
int CudaSearch<T>::getSpaceDim() {
    return spaceDim;
}

template <typename T>
CudaSearch<T>::CudaSearch(const std::vector< std::vector<T> > &dataset) : datasetSize(dataset.size()){

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

template <typename T>
__global__ void _cudaDistances(const T *__restrict__ dataset, const T *__restrict__ query,
                               int *__restrict__ nnIndexes, T *__restrict__ nnDistancesSqr, const int datasetSize,
                               const int spaceDim){//, std::vector<int> v){
    int i;
    i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(i >= datasetSize)
        return;
    T dist = 0;
    int j;
    for(j=0; j < spaceDim; j++){
        const T diff = query[j] - dataset[i*spaceDim + j];
        dist = dist + (diff * diff);
    }
    // TODO vedere se è possibile modificare questa implementazione per velocizzare il calcolo delle distanze

//  loop unrolling
//    T* ptrLastQuery = query + spaceDim;
//    T* ptrLastQueryGroup = ptrLastQuery - 3;
//    dataset = dataset + (i*spaceDim);
//    while (query < ptrLastQueryGroup) {
//        const T diff0 = query[0] - dataset[0];
//        const T diff1 = query[1] - dataset[1];
//        const T diff2 = query[2] - dataset[2];
//        const T diff3 = query[3] - dataset[3];
//        const T diff4 = query[4] - dataset[4];
//        const T diff5 = query[5] - dataset[5];
//        const T diff6 = query[6] - dataset[6];
//        const T diff7 = query[7] - dataset[7];
//        dist = dist + (diff0 * diff0) + (diff1 * diff1) + (diff2 * diff2) + (diff3 * diff3) + (diff4 * diff4) + (diff5 * diff5) + (diff6 * diff6) + (diff7 * diff7);
//        query = query + 8;
//        dataset = dataset + 8;
//    }
////    while (query < ptrLastQueryGroup) {
////        const T diff = query[0] - dataset[0];
////        dist = dist + (diff*diff);
////        query += 1;
////        dataset += 1;
////    }

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
    // TODO capire il numero ottimale della dimensione dei blocchi dal file excel di NVIDIA
    int numBlocks = static_cast<int>((datasetSize+BLOCK_SIZE-1) / BLOCK_SIZE);
    _cudaDistances <<<numBlocks,BLOCK_SIZE>>>(this->dataset, this->query, this->nnIndexes, this->nnDistancesSqr, this->datasetSize, spaceDim);//, v);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // sort by increasing distance
    thrust::device_ptr<T> device_thrustDistances(this->nnDistancesSqr);
    thrust::device_ptr<int> device_thrustIndexes(this->nnIndexes);
    thrust::sort_by_key(device_thrustDistances, device_thrustDistances+this->datasetSize, device_thrustIndexes);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // TODO questo serve?

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
