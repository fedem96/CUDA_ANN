#ifndef CUDA_NANOFLANN_SEARCH_HPP
#define CUDA_NANOFLANN_SEARCH_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cstring>
//#include <math_constants.h>
//#include <Eigen>
#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"

template <typename T>
class Search{
public:
    virtual void search(const std::vector<T> &query, std::vector<size_t> &nnIndexes, std::vector<T> &nnDistancesSqr, const size_t &numResults) = 0;
    virtual ~Search() {};
};

template <typename T>
class CpuSearch : public Search<T> {
public:
    CpuSearch(const std::vector< std::vector<T> > &dataset, const bool deepcopy=false);
    void search(const std::vector<T> &query, std::vector<size_t> &nnIndexes, std::vector<T> &nnDistancesSqr, const size_t &numResults) override;
    virtual ~CpuSearch();
private:
    //T* dataset;
    bool deepcopy;
    KDTreeVectorOfVectorsAdaptor< std::vector<std::vector<T> >, T > kdTree;

    //std::vector< std::vector<T> > dataset;
};


template<typename T>
CpuSearch<T>::CpuSearch(const std::vector< std::vector<T> > &dataset, const bool deepcopy) :
deepcopy(deepcopy),
kdTree(dataset[0].size(), dataset, 10 /* max leaf */ ) // occhio a max leaf!
{
//    if(deepcopy){
//        this->dataset = new T[DATASET_SIZE*SPACE_DIM];
//        std::memcpy(this->dataset, dataset, DATASET_SIZE*SPACE_DIM);
//    } else {
//        this->dataset = dataset;
//    }

    /* prova */
    //this->dataset = dataset;
}

template<typename T>
void CpuSearch<T>::
search(const std::vector<T> &query, std::vector<size_t> &nnIndexes, std::vector<T> &nnDistancesSqrt, const size_t &numResults){


    // construct a kd-tree index:
    // Dimensionality set at run-time (default: L2)
    // ------------------------------------------------------------

//    int elementSize = query.size();
//    size_t datasetSize = dataset.size();
//    T* data = new T[datasetSize*elementSize];
//    for(size_t i=0; i < datasetSize; i++)
//        std::memcpy(&data[i*elementSize], &dataset[i][0], sizeof(T) * elementSize);
//    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> mat(T);
//    delete [] data;
//    typedef nanoflann::KDTreeEigenMatrixAdaptor< Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > MyKDTree_t;
//    const int max_leaf = 10;
////    MyKDTree_t mat_index(mat, max_leaf);
////    mat_index.index->buildIndex();

    // do a knn search
    /* funziona */
    nanoflann::KNNResultSet<T> resultSet(numResults);
    resultSet.init(&nnIndexes[0], &nnDistancesSqrt[0] );
    kdTree.index->findNeighbors(resultSet, &query[0], nanoflann::SearchParams(50)); // TODO vedere searchparams che fa

    /* prova */
//    size_t datasetSize = dataset.size();
//    int elementSize = query.size();
//    T* data = new T[datasetSize*elementSize];
//    for(size_t i=0; i < datasetSize; i++)
//        std::memcpy(&data[i*elementSize], &dataset[i][0], sizeof(T) * elementSize);
//    Eigen::Matrix<T,
//    Eigen::Dynamic,
//    128, Eigen::RowMajor> mat(datasetSize, elementSize);
//    for(size_t i=0; i < datasetSize; i++)
//        for(int j=0; j < elementSize; j++)
//            mat << dataset[i][j];
//    //=Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> > (data, datasetSize, elementSize);
//    //delete [] data;
//    typedef nanoflann::KDTreeEigenMatrixAdaptor< Eigen::Matrix<T,Eigen::Dynamic,128, Eigen::RowMajor> > MyKDTree_t;
//    const int max_leaf = 10;
//
//    MyKDTree_t mat_index(query.size(), mat, max_leaf);
//    //mat_index.index->buildIndex(); mat_index.index->... \endcode
//    //nanoflann::KNNResultSet<T> resultSet(numResults);
//    resultSet.init(&nnIndexes[0], &nnDistancesSqrt[0] );
//    mat_index.index->findNeighbors(resultSet, &query[0], nanoflann::SearchParams(10));
//    printf("debug");
}

template<typename T>
CpuSearch<T>::~CpuSearch() {
//    if(this->deepcopy)
//        delete [] this->dataset;
}


#endif //CUDA_NANOFLANN_SEARCH_HPP