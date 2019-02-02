/*
 ============================================================================
 Name        : CUDA_ANN.cu
 Author      : federico
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA Approximate Nearest Neighbors
 ============================================================================
 */

#include <string>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <vector>

#include "cuda.h"

#include "read_sift_dataset.cpp"

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

template<class T>
std::string vecToStr(std::vector<T> vec) {
	std::ostringstream oss;

	if (!vec.empty())
	{
		// Convert all but the last element to avoid a trailing ","
		std::copy(vec.begin(), vec.end()-1, std::ostream_iterator<T>(oss, ","));

		// Now add the last element with no delimiter
		oss << vec.back();
	}

	return oss.str();
}

__global__ void greet(){
	std::cout << "ihjbg";
}

int main(void)
{
	std::string baseFileName = "data/siftsmall/siftsmall_base.fvecs";
	std::string groundtruthFileName = "data/siftsmall/siftsmall_groundtruth.ivecs";
	std::string queryFileName = "data/siftsmall/siftsmall_query.fvecs";

	std::vector< std::vector<float> > baseVec;
	bool result = readVecsFile<float,float>(baseFileName, baseVec, false);
	// TODO check result
	std::vector< std::vector<int> > grthVec;
	result = readVecsFile<int,int>(groundtruthFileName, grthVec, false);
	std::vector< std::vector<float> > queryVec;
	result = readVecsFile<float,float>(queryFileName, queryVec, false);


	std::cout << std::endl;
	std::cout << "base dataset size:\n\t" << baseVec.size() << std::endl;
	std::cout << "groundtruth dataset size:\n\t" << grthVec.size() << std::endl;
	std::cout << "query dataset size:\n\t" << queryVec.size() << std::endl;
	
	std::cout << std::endl;
	std::cout << "element 0 of vector base (" << baseVec[0].size() << " elements):\n\t" << vecToStr<float>(baseVec[0]) << std::endl;
	std::cout << "element 0 of vector groundtruth (" << grthVec[0].size() << " elements):\n\t" << vecToStr<int>(grthVec[0]) << std::endl;
	std::cout << "element 0 of vector query (" << queryVec[0].size() << " elements):\n\t" << vecToStr<float>(queryVec[0]) << std::endl;
	std::cout << std::endl;
	std::cout << "element 13 of vector base (" << baseVec[13].size() << " elements):\n\t" << vecToStr<float>(baseVec[13]) << std::endl;
	std::cout << "element 13 of vector groundtruth (" << grthVec[13].size() << " elements):\n\t" << vecToStr<int>(grthVec[13]) << std::endl;
	std::cout << "element 13 of vector query (" << queryVec[13].size() << " elements):\n\t" << vecToStr<float>(queryVec[13]) << std::endl;
	
	std::cout << std::endl;
	// TODO capire come è fatto il groundtruth (data una query, quanti elementi ci sono nel groundtruth?)
	// TODO far funzionare il debug
	std::cout << "groundtruth sizeeeeeeeeeee: " << sizeof(grthVec) << " bytes" << std::endl;
	std::cout << "groundtruth element size: " << sizeof(grthVec[13]) << " bytes" << std::endl;

	greet<<<1,10>>>();
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	// TODO provare libreria FLANN o nanoFLANN in versione sequenziale
	return 0;
}

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

