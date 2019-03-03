#ifndef CUDA_NANOFLANN_SEARCH_CUH
#define CUDA_NANOFLANN_SEARCH_CUH

#include <cuda.h>
#include <thrust/sort.h>
#include "search.hpp"
#define BLOCK_SIZE 1024

//#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define CUDA_CHECK_RETURN(value) { gpuAssert((value), __FILE__, __LINE__); }
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);

template <typename T>
class CudaSearch : public Search<T> {
public:
    CudaSearch(const std::vector< std::vector<T> > &dataset);
    void search(const std::vector<T> &host_query, std::vector<size_t> &host_nnIndexes, std::vector<T> &host_nnDistancesSqr, const size_t &numResults) override;
    virtual ~CudaSearch();
private:
    T* dataset;
    T* query;
    size_t* nnIndexes;
    T* nnDistancesSqr;
    size_t datasetSize;
};


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
    exit (1);
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template <typename T>
CudaSearch<T>::CudaSearch(const std::vector< std::vector<T> > &dataset) : datasetSize(dataset.size()){

    assert(datasetSize > 0);

    // memory allocation on device
    CUDA_CHECK_RETURN(
            cudaMalloc((void ** )&this->dataset, sizeof(T) * datasetSize * dataset[0].size())
    );
    CUDA_CHECK_RETURN(
            cudaMalloc((void ** )&this->query, sizeof(T) * dataset[0].size())
    );
    CUDA_CHECK_RETURN(
            cudaMalloc((void ** )&this->nnIndexes, sizeof(size_t) * datasetSize)
    );
    CUDA_CHECK_RETURN(
            cudaMalloc((void ** )&this->nnDistancesSqr, sizeof(T) * datasetSize)
    );

    // dataset copy to device
    for(size_t i=0; i < datasetSize; i++){
        CUDA_CHECK_RETURN(
                cudaMemcpy(&this->dataset[i*dataset[0].size()], &dataset[i][0], sizeof(T) * dataset[0].size(), cudaMemcpyHostToDevice)
        );
    }

}

template <typename T>
__global__ void _cudaSearch(const T* __restrict__ dataset, const T* __restrict__ query, size_t* __restrict__ nnIndexes, T* __restrict__ nnDistancesSqr, const size_t datasetSize, const int spaceDim){
    size_t i;
    i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(i >= datasetSize)
        return;
    T dist = 0;
    int j;
    for(j=0; j < spaceDim; j++){
        const T diff = query[j] - dataset[i*spaceDim + j];
        dist = dist + (diff * diff);
    }
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

template <typename T>
void CudaSearch<T>::
search(const std::vector<T> &host_query, std::vector<size_t> &host_nnIndexes, std::vector<T> &host_nnDistancesSqr, const size_t &numResults){


    CUDA_CHECK_RETURN(
            cudaMemcpy(this->query, &host_query[0], sizeof(T) * host_query.size(), cudaMemcpyHostToDevice)
    );

    int numBlocks = static_cast<int>((datasetSize+BLOCK_SIZE-1) / BLOCK_SIZE);
    _cudaSearch<<<numBlocks,BLOCK_SIZE>>>(this->dataset, this->query, this->nnIndexes, this->nnDistancesSqr, this->datasetSize, host_query.size());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    thrust::device_ptr<T> device_thrustDistances(this->nnDistancesSqr);
    thrust::device_ptr<size_t> device_thrustIndexes(this->nnIndexes);
    thrust::sort_by_key(device_thrustDistances, device_thrustDistances+this->datasetSize, device_thrustIndexes);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // copy from device to host memory
    CUDA_CHECK_RETURN(
            cudaMemcpy(&host_nnIndexes[0], this->nnIndexes, sizeof(size_t) * numResults, cudaMemcpyDeviceToHost)
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

#endif //CUDA_NANOFLANN_SEARCH_CUH
