#ifndef CUDA_SEARCH_HPP
#define CUDA_SEARCH_HPP

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include <cstring>
#include <thrust/sort.h>

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
    CpuSearch(T* dataset, int rows, int cols, int numCores=0);
    CpuSearch(const std::vector< std::vector<T> > &dataset, int numCores=0);
    void search(T* query, std::vector<int> &nnIndexes, std::vector<T> &nnDistancesSqr, const int &numResults) override;
    int getSpaceDim() override;
    virtual ~CpuSearch();
private:
    T* dataset;
    int datasetSize;
    int spaceDim;
    bool datasetAllocated;
    int numCores;

    std::vector<int> nnAllIndexes;
    std::vector<T> nnAllDistancesSqr;

};

// TODO modificare costruttore: permettere di scegliere se si vuole la versione sequenziale oppure OpenMP (nel secondo caso, permettere di scegliere il numero di thread)
template<typename T>
CpuSearch<T>::CpuSearch(T *dataset, int datasetSize, int spaceDim, int numCores) : dataset(dataset), datasetSize(datasetSize), spaceDim(spaceDim), datasetAllocated(false), numCores(numCores), nnAllIndexes(datasetSize), nnAllDistancesSqr(datasetSize) {
    assert(datasetSize > 0);
}

template<typename T>
CpuSearch<T>::CpuSearch(const std::vector< std::vector<T> > &dataset_vv, int numCores) : datasetSize(dataset_vv.size()), numCores(numCores), nnAllIndexes(dataset_vv.size()), nnAllDistancesSqr(dataset_vv.size())
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
T compare (const void * a, const void * b)
{
    return ( *(T*)a - *(T*)b );
}

template<typename T>
void CpuSearch<T>::search(T* query, std::vector<int> &nnIndexes, std::vector<T> &nnDistancesSqr, const int &numResults){
    // TODO finire di implementare la ricerca, versione cpu sequenziale / OpenMP

    T dist, diff;
    T* k = dataset;
    for(int i=0; i < datasetSize ; i++) {
        dist = 0;
        for (int j = 0; j < spaceDim; j++) {
            // TODO provare loop unrolling
            diff = query[j] - *k;
            dist = dist + (diff * diff);
            k++;
        }

        // dist is the square of the distance: for efficiency I use it instead of distance to find neighbors
        nnAllDistancesSqr[i] = dist;
        nnAllIndexes[i] = i;
    }

    // sorting whit thrust is expensive why?
    // sort by increasing distance
    thrust::sort_by_key(&nnAllDistancesSqr[0], &nnAllDistancesSqr[0]+this->datasetSize, &nnAllIndexes[0]);
    // TODO usare std::sort o qsort che sono più veloci. Però c'è da fare in modo che venga ordinato anche il vettore degli indici
    //std::sort(&nnAllDistancesSqr[0], &nnAllDistancesSqr[0]+this->datasetSize);
    //qsort (&nnAllDistancesSqr[0], this->datasetSize, sizeof(T), compare);

    // copy distances and indexes of nearest neighbors
    std::memcpy(&nnDistancesSqr[0], &nnAllDistancesSqr[0], sizeof(T)  * numResults);
    std::memcpy(&nnIndexes[0], &nnAllIndexes[0], sizeof(int) * numResults);
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