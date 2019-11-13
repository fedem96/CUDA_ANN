#ifndef CUDA_SEARCH_HPP
#define CUDA_SEARCH_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cstring>
#include <flann/flann.hpp>
//#include <math_constants.h>

//using namespace flann;

template <typename T>
class Search{
public:
    virtual void search(T* query, std::vector<size_t> &nnIndexes, std::vector<T> &nnDistancesSqr, const size_t &numResults) = 0;
    virtual int getSpaceDim() = 0;
    virtual ~Search() {};
};

template <typename T>
class CpuSearch : public Search<T> {
public:
    CpuSearch(T* dataset, int rows, int cols, const bool deepcopy=false);
    CpuSearch(const std::vector< std::vector<T> > &dataset, const bool deepcopy=false);
    void search(T* query, std::vector<size_t> &nnIndexes, std::vector<T> &nnDistancesSqr, const size_t &numResults) override;
    int getSpaceDim() override;
    virtual ~CpuSearch();
private:
    flann::Matrix<T> dataset;
    bool deepcopy;

    //std::vector< std::vector<T> > dataset;
};


template<typename T>
CpuSearch<T>::CpuSearch(T *dataset, int rows, int cols, const bool deepcopy) :
dataset(dataset, rows, cols)
{
//    if(deepcopy){
//        T* tmpDataset = new T[rows*cols];
//        std::memcpy(tmpDataset, dataset, rows*cols);
//        dataset = tmpDataset;
//    }

}

/*template<typename T>
CpuSearch<T>::CpuSearch(const std::vector< std::vector<T> > &dataset, const bool deepcopy) :
dataset(vvToPtr(dataset), dataset.size(), dataset[0].size()),
deepcopy(deepcopy) // occhio a max leaf!
{
//    if(deepcopy){
//        this->dataset = new T[DATASET_SIZE*SPACE_DIM];
//        std::memcpy(this->dataset, dataset, DATASET_SIZE*SPACE_DIM);
//    } else {
//        this->dataset = dataset;
//    }

}*/

template<typename T>
void CpuSearch<T>::
search(T* query, std::vector<size_t> &nnIndexes, std::vector<T> &nnDistancesSqr, const size_t &numResults){
    flann::Matrix<T> query_mat(query, 1, this->dataset.cols);
    // TODO fare la ricerca per davvero
}

template<typename T>
int CpuSearch<T>::getSpaceDim() {
    return dataset.cols;
}

template<typename T>
CpuSearch<T>::~CpuSearch() {
//    if(this->deepcopy)
//        delete [] this->dataset;
}



#endif //CUDA_SEARCH_HPP