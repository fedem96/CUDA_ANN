#ifndef CUDA_SEARCH_HPP
#define CUDA_SEARCH_HPP

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include <cstring>
// #include <thrust/sort.h>

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
};

// TODO modificare costruttore: permettere di scegliere se si vuole la versione sequenziale oppure OpenMP (nel secondo caso, permettere di scegliere il numero di thread)
template<typename T>
CpuSearch<T>::CpuSearch(T *dataset, int datasetSize, int spaceDim, int numCores) : dataset(dataset), datasetSize(datasetSize), spaceDim(spaceDim), datasetAllocated(false), numCores(numCores) {}

template<typename T>
CpuSearch<T>::CpuSearch(const std::vector< std::vector<T> > &dataset_vv, int numCores) : datasetSize(dataset_vv.size()), numCores(numCores)
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
    // TODO implementare la ricerca, versione cpu sequenziale / OpenMP

    std::vector<int> nnAllIndexes(datasetSize); // resize dei vetteri per ordinamento ALL
    std::vector<T> nnAllDistancesSqr(datasetSize);

    for(int i=0; i < datasetSize ; i++) {
        T dist = 0;
        for (int j = 0; j < spaceDim; j++) {
            const T diff = query[j] - dataset[i * spaceDim + j];
            dist = dist + (diff * diff);
        }
        nnAllDistancesSqr[i] = dist; // sarebbe sotto radice ma è uguale
        nnAllIndexes[i] = i;
    }

    // sorting whit thrust is expensive why?
    // thrust::sort_by_key(&nnAllDistancesSqr[0], &nnAllDistancesSqr[0]+this->datasetSize, &nnAllIndexes[0]);
    // TODO usare std::sort o qsort che sono più veloci. Però c'è da fare in modo che venga ordinato anche il vettore degli indici
    // std::sort(&nnAllDistancesSqr[0], &nnAllDistancesSqr[0]+this->datasetSize);
    qsort (&nnAllDistancesSqr[0], this->datasetSize, sizeof(T), compare);

    // copia dei primi 100 elementi
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