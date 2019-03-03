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

void dataPrint(std::vector< std::vector<float> > &, std::vector< std::vector<int> > &, std::vector< std::vector<float> > &);

template <typename T>
std::chrono::duration<double> evaluate(Search<T> *s, std::vector< std::vector<T> > &queryVec, std::vector< std::vector<int> > &grthVec, int &numResults, bool mustCheck=false);

template <typename T>
void check(std::vector< std::vector<int> > &grthVec, std::vector<std::vector<size_t> > &nnAllIndexes, std::vector<std::vector<T> > &nnAllDistancesSqr);

int main(int argc, char **argv)
{
    /* files path  definition*/
//	std::string baseFileName = "../data/sift/sift_base.fvecs";
//	std::string groundtruthFileName = "../data/sift/sift_groundtruth.ivecs";
//	std::string queryFileName = "../data/sift/sift_query.fvecs";
	std::string baseFileName = "../data/siftsmall/siftsmall_base.fvecs";
	std::string groundtruthFileName = "../data/siftsmall/siftsmall_groundtruth.ivecs";
	std::string queryFileName = "../data/siftsmall/siftsmall_query.fvecs";

	/* data structures for dataset */
	std::vector< std::vector<float> > host_dataset;
	std::vector< std::vector<int> > grthVec;
	std::vector< std::vector<float> > queryVec;

	std::cout << "Reading dataset" << std::endl;
	assert((readVecsFile<float,float>(baseFileName, host_dataset, true)));
    std::cout << "\nReading queries" << std::endl;
    assert((readVecsFile<float,float>(queryFileName, queryVec, true)));
    std::cout << "\nReading groundtruth for queries" << std::endl;
	assert((readVecsFile<int,int>(groundtruthFileName, grthVec, true)));

    //host_dataset = std::vector< std::vector<float> >(host_dataset.begin(), host_dataset.begin() + 10000);
    host_dataset.resize(10000);

	const size_t datasetSize = host_dataset.size();
    assert(queryVec.size() == grthVec.size());

    int numResults = 20;

    /* some print to understand data */
    dataPrint(host_dataset, grthVec, queryVec);

//    for(int i=0; i<grthVec.size(); i++)
//        for(int j=0; j<grthVec[0].size(); j++)
//            if(!(grthVec[i][j] >= 0 && grthVec[i][j] < datasetSize))
//            	cout << "ERRORE: "<< i << " " << j<< " "<< grthVec[i][j] << std::endl;


	//// CPU evaluation
	Search<float> *s = new CpuSearch<float>(host_dataset);
	//std::chrono::duration<double> cpuTime = evaluate<float>(s, queryVec, grthVec, numResults, true);
    //std::cout << "CPU duration: " << cpuTime.count() << std::endl;
	delete s;

	//// GPU evaluation
	s = new CudaSearch<float>(host_dataset);
    std::chrono::duration<double> gpuTime = evaluate<float>(s, queryVec, grthVec, numResults, false);
    std::cout << "GPU duration: " << gpuTime.count() << std::endl;
    delete s;

//    double speedup = cpuTime.count() / gpuTime.count();
//    std::cout << "Speedup: " << speedup << std::endl;

	return 0;
}

template <typename T>
std::chrono::duration<double> evaluate(Search<T> *s, std::vector< std::vector<T> > &queryVec, std::vector< std::vector<int> > &grthVec, int &numResults, bool mustCheck){

    int numQueries = queryVec.size();

    /* data structures for query */
    std::vector<std::vector<size_t> > nnAllIndexes(numQueries, std::vector<size_t>(numResults));
    std::vector<std::vector<T> > nnAllDistancesSqr(numQueries, std::vector<float>(numResults));
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < numQueries; i++){
        s->search(queryVec[i], nnAllIndexes[i], nnAllDistancesSqr[i], numResults);
        //std::cout << "\nelement " << i << " of vector nnAllDistancesSqrt (" << nnAllDistancesSqrt[i].size() << " elements):\n\t" << vecToStr<float>(nnAllDistancesSqrt[i]) << std::endl;
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



void dataPrint(std::vector< std::vector<float> > &baseVec, std::vector< std::vector<int> > &grthVec, std::vector< std::vector<float> > &queryVec){

    std::cout << std::endl;
    std::cout << "base dataset size:\n\t" << baseVec.size() << std::endl;
    std::cout << "groundtruth dataset size:\n\t" << grthVec.size() << std::endl;
    std::cout << "query dataset size:\n\t" << queryVec.size() << std::endl;

    std::cout << std::endl;
    std::cout << "ELEMENT 0" << std::endl;
    std::cout << "\nelement 0 of vector base (" << baseVec[0].size() << " elements):\n\t" << vecToStr<float>(baseVec[0]) << std::endl;
    std::cout << "\nelement 0 of vector groundtruth (" << grthVec[0].size() << " elements):\n\t" << vecToStr<int>(grthVec[0]) << std::endl;
    std::cout << "\nelement 0 of vector query (" << queryVec[0].size() << " elements):\n\t" << vecToStr<float>(queryVec[0]) << std::endl;
    std::cout << std::endl;
    std::cout << "ELEMENT 13" << std::endl;
    std::cout << "\nelement 13 of vector base (" << baseVec[13].size() << " elements):\n\t" << vecToStr<float>(baseVec[13]) << std::endl;
    std::cout << "\nelement 13 of vector groundtruth (" << grthVec[13].size() << " elements):\n\t" << vecToStr<int>(grthVec[13]) << std::endl;
    std::cout << "\nelement 13 of vector query (" << queryVec[13].size() << " elements):\n\t" << vecToStr<float>(queryVec[13]) << std::endl;

    std::cout << std::endl;
    std::cout << "SIZES" << std::endl;
    std::cout << "\ngroundtruth sizeof: " << sizeof(grthVec) << " bytes" << std::endl;
    std::cout << "groundtruth element sizeof: " << sizeof(grthVec[13]) << " bytes" << std::endl;
}