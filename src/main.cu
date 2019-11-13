/*
 ============================================================================
 Name        : CUDA_ANN.cu
 Author      : federico
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA Approximate Nearest Neighbors
 ============================================================================
 */

//#include <string>
//#include <algorithm>
#include <sstream>
#include <iterator>
#include <iostream>
//#include <numeric>
//#include <stdlib.h>
#include <vector>
#include <chrono>

#include <cuda.h>

#include "utils.cpp"
#include "read_sift_dataset.cpp"
#include "search.hpp"
#include "search.cuh"

template <typename T>
std::chrono::duration<double> evaluate(Search<T> *s, T* queries, std::vector <std::vector<int>> &grthVec, const int &numQueries, int &numResults, bool mustCheck = false);

template <typename T>
std::chrono::duration<double> oldEvaluate(Search<T> *s, std::vector <std::vector<T>> &queryVec,
                                          std::vector <std::vector<int>> &grthVec, int &numResults,
                                          bool mustCheck = false);

template <typename T>
void check(std::vector< std::vector<int> > &grthVec, std::vector<std::vector<size_t> > &nnAllIndexes, std::vector<std::vector<T> > &nnAllDistancesSqr);

int main(int argc, char **argv)
{
    /* files path definition */
    // 10^6 examples dataset:
//	std::string baseFileName = "../data/sift/sift_base.fvecs";
//	std::string groundtruthFileName = "../data/sift/sift_groundtruth.ivecs";
//	std::string queryFileName = "../data/sift/sift_query.fvecs";
	// 10^4 examples dataset:
	std::string baseFileName = "../data/siftsmall/siftsmall_base.fvecs";
	std::string groundtruthFileName = "../data/siftsmall/siftsmall_groundtruth.ivecs";
	std::string queryFileName = "../data/siftsmall/siftsmall_query.fvecs";

	/* evaluation parameters */
    int numResults = 100;

	/* data structures for dataset */
	std::vector< std::vector<float> > host_dataset_vv;      // datasetSize x spaceDim       <- dataset where to find nearest neighbors
    std::vector< std::vector<float> > host_queries_vv;      // numQueries  x spaceDim       <- test samples
	std::vector< std::vector<int> > host_grTruth_vv;        // numQueries  x 100            <- first 100 nearest neighbors for each test sample

	/* reading of dataset, queries and groundtruth */
	std::cout << "Reading dataset" << std::endl;
	assert((readVecsFile<float,float>(baseFileName, host_dataset_vv, false)));
    std::cout << "\nReading queries" << std::endl;
    assert((readVecsFile<float,float>(queryFileName, host_queries_vv, false)));
    std::cout << "\nReading groundtruth for queries" << std::endl;
	assert((readVecsFile<int,int>(groundtruthFileName, host_grTruth_vv, false)));

	// dataset slice (to do quick tests)
    //host_dataset_vv = std::vector< std::vector<float> >(host_dataset_vv.begin(), host_dataset_vv.begin() + 10000);
    //host_dataset_vv.resize(10000);

    /* constants initialization */
	const size_t datasetSize = host_dataset_vv.size();
	const int spaceDim = host_dataset_vv[0].size();     // 128
	const int numQueries = host_queries_vv.size();
    assert(host_queries_vv.size() == host_grTruth_vv.size());          // host_queries_vv and host_grTruth_vv must have same length
    assert(numResults <= host_grTruth_vv[0].size());            // assert(numResults <= 100)

    /* some print to understand data */
    //dataPrint(host_dataset_vv, host_grTruth_vv, host_queries_vv);
//    for(int i=0; i<host_grTruth_vv.size(); i++)
//        for(int j=0; j<host_grTruth_vv[0].size(); j++)
//            if(!(host_grTruth_vv[i][j] >= 0 && host_grTruth_vv[i][j] < datasetSize))
//            	cout << "ERRORE: "<< i << " " << j<< " "<< host_grTruth_vv[i][j] << std::endl;


    /* data conversion */
    // convert queries from vector of vectors into raw pointer
    // TODO capire perché con la pinned va più lento mentre invece dovrebbe essere più veloce
    float* host_queries_ptr;
//    CUDA_CHECK_RETURN(
//            cudaMallocHost((void ** )&host_queries_ptr, sizeof(float) * numQueries * spaceDim)     // allocate pinned memory on host RAM: it allows the use of DMA, speeding up cudaMemcpy
//    );
    host_queries_ptr = new float[numQueries * spaceDim]; // non-pinned memory
    for(size_t i=0; i < numQueries; i++){
        std::memcpy(host_queries_ptr + (i*spaceDim), &host_queries_vv[i][0], sizeof(float) * spaceDim);
    }

    /* evaluation */
    Search<float> *s;

	//// CPU evaluation
	s = new CpuSearch<float>(host_dataset_ptr, datasetSize, spaceDim);
	std::chrono::duration<double> cpuEvalTime = evaluate<float>(s, host_queries_ptr, host_grTruth_vv, numQueries, numResults, false);
    std::cout << "CPU eval time: " << cpuEvalTime.count() << std::endl;
	delete s;

	//// GPU evaluation
    auto start = std::chrono::high_resolution_clock::now();
	s = new CudaSearch<float>(host_dataset_vv);
    std::chrono::duration<double> gpuInitTime = std::chrono::high_resolution_clock::now() - start;
    std::chrono::duration<double> gpuEvalTime = evaluate<float>(s, host_queries_ptr, host_grTruth_vv, numQueries, numResults, false);
    std::cout << "GPU init time: " << gpuInitTime.count() << std::endl;
    std::cout << "GPU eval time: " << gpuEvalTime.count() << std::endl;
    delete s;

//    double speedup = cpuEvalTime.count() / gpuEvalTime.count();
//    std::cout << "Speedup: " << speedup << std::endl;

    //delete [] host_dataset_ptr;

	return 0;
}

template <typename T>
std::chrono::duration<double> evaluate(Search<T> *s, T* queries_ptr, std::vector <std::vector<int>> &grthVec, const int &numQueries, int &numResults, bool mustCheck){

    /* data structures for query */
    std::vector<std::vector<size_t> > nnAllIndexes(numQueries, std::vector<size_t>(numResults));    // numQueries x numResults
    std::vector<std::vector<T> > nnAllDistancesSqr(numQueries, std::vector<float>(numResults));     // numQueries x numResults

    auto start = std::chrono::high_resolution_clock::now();
    int sd = s->getSpaceDim();
    for(int i = 0; i < numQueries; i++){
        s->search(&queries_ptr[i * sd], nnAllIndexes[i], nnAllDistancesSqr[i], numResults);
        //std::cout << vecToStr(nnAllDistancesSqr[i]) << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;


    if(mustCheck)
        check(grthVec, nnAllIndexes, nnAllDistancesSqr);

    return diff;
}

template <typename T>
void check(std::vector< std::vector<int> > &grthVec, std::vector<std::vector<size_t> > &nnAllIndexes, std::vector<std::vector<T> > &nnAllDistancesSqr){
    for(int i = 0; i < nnAllIndexes.size(); i++){
        //std::cout << "\nelement " << i << " of vector groundtruth (" << grthVec[i].size() << " elements):\n\t" << vecToStr<int>(grthVec[i]) << std::endl;
        //std::cout << "\nelement " << i << " of vector nnAllIndexes (" << nnAllIndexes[i].size() << " elements):\n\t" << vecToStr<size_t>(nnAllIndexes[i]) << std::endl;
        checkKNN<float>(grthVec[i], nnAllIndexes[i], nnAllDistancesSqr[i]);
    }
}