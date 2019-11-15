#ifndef CUDA_SEARCH_HPP
#define CUDA_SEARCH_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cstring>

template <typename T>
class Search{
public:
    virtual void search(T* query, std::vector<int> &nnIndexes, std::vector<T> &nnDistancesSqr, const int &numResults) = 0;
    virtual int getSpaceDim() = 0;
    virtual ~Search() {};
};

template <typename T>
class CpuSearch : public Search<T> {
public:
    CpuSearch(T* dataset, int rows, int cols);
    CpuSearch(const std::vector< std::vector<T> > &dataset);
    void search(T* query, std::vector<int> &nnIndexes, std::vector<T> &nnDistancesSqr, const int &numResults) override;
    int getSpaceDim() override;
    virtual ~CpuSearch();
private:
    T* dataset;
    int datasetSize;
    int spaceDim;
    bool datasetAllocated;
};

// TODO modificare costruttore: permettere di scegliere se si vuole la versione sequenziale oppure OpenMP (nel secondo caso, permettere di scegliere il numero di thread)

template<typename T>
CpuSearch<T>::CpuSearch(T *dataset, int datasetSize, int spaceDim) : dataset(dataset), datasetSize(datasetSize), spaceDim(spaceDim), datasetAllocated(false) {}

template<typename T>
CpuSearch<T>::CpuSearch(const std::vector< std::vector<T> > &dataset_vv) : datasetSize(dataset_vv.size())
{
    assert(datasetSize > 0);
    spaceDim = dataset_vv[0].size();
    dataset = new T[datasetSize * spaceDim];
    datasetAllocated = true;
    for(int i=0; i < datasetSize; i++){
        std::memcpy(dataset + (i*spaceDim), &dataset_vv[i][0], sizeof(T) * spaceDim);
    }
}

template<typename T>
void CpuSearch<T>::
search(T* query, std::vector<int> &nnIndexes, std::vector<T> &nnDistancesSqr, const int &numResults){
    // TODO implementare la ricerca, versione cpu sequenziale / OpenMP
}

template<typename T>
int CpuSearch<T>::getSpaceDim() {
    return spaceDim;
}

template<typename T>
CpuSearch<T>::~CpuSearch() {
    if(datasetAllocated)
        delete [] dataset;
}



#endif //CUDA_SEARCH_HPP