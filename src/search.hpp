#ifndef CUDA_SEARCH_HPP
#define CUDA_SEARCH_HPP

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include <cstring>


template<typename T>
struct Neighbor {
    T distance;
    int index;
    bool operator>(const Neighbor& rhs) { return distance > rhs.distance; }
    bool operator>=(const Neighbor& rhs) { return distance >= rhs.distance; }
    bool operator<(const Neighbor& rhs) { return distance < rhs.distance; }
    bool operator<=(const Neighbor& rhs) { return distance <= rhs.distance; }
    bool operator==(const Neighbor& rhs) { return distance == rhs.distance; }
};

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
    std::vector<Neighbor<T> > nearestNeighbors;

    T* dists;

};

// TODO the contructor must specify the number of cores for the openMP version
template<typename T>
CpuSearch<T>::CpuSearch(T *dataset, int datasetSize, int spaceDim, int numCores) : dataset(dataset), datasetSize(datasetSize), spaceDim(spaceDim), datasetAllocated(false), numCores(numCores), nnAllIndexes(datasetSize), nnAllDistancesSqr(datasetSize), nearestNeighbors(datasetSize) {
    assert(datasetSize > 0);
    dists = new T[spaceDim];
}

template<typename T>
CpuSearch<T>::CpuSearch(const std::vector< std::vector<T> > &dataset_vv, int numCores) : datasetSize(dataset_vv.size()), numCores(numCores), nnAllIndexes(dataset_vv.size()), nnAllDistancesSqr(dataset_vv.size()), nearestNeighbors(dataset_vv.size())
{
    assert(datasetSize > 0);
    spaceDim = dataset_vv.at(0).size();
    dataset = new T[datasetSize * spaceDim];
    datasetAllocated = true;
    for(int i=0; i < datasetSize; i++){
        std::memcpy(dataset + (i*spaceDim), &dataset_vv[i][0], sizeof(T) * spaceDim);
    }
    dists = new T[spaceDim];
}

template<typename T>
T comparator(const void * a, const void * b){
    return ((Neighbor<T> *)a)->distance - ((Neighbor<T> *)b)->distance;
}

template<typename T>
void CpuSearch<T>::search(T* query, std::vector<int> &nnIndexes, std::vector<T> &nnDistancesSqr, const int &numResults){

    omp_set_num_threads(numCores);
    #pragma omp parallel for
    for(int i=0; i < datasetSize ; i++) {
        T dist = 0;
        T* dataset_tmp = dataset + (i*spaceDim);
        for (int j = 0; j < spaceDim; j++) {
            const T diff = query[j] - dataset_tmp[j];
            dist += diff * diff;
        }

        // dist is the square of the distance: for efficiency I use it instead of distance to find neighbors
        nearestNeighbors[i].distance = dist;
        nearestNeighbors[i].index = i;
    }

    // sort by increasing distance
    qsort (&nearestNeighbors[0], this->datasetSize, sizeof(Neighbor<T>), comparator);

    // copy distances and indexes of nearest neighbors
    for(int i = 0; i < numResults; i++) {
        std::memcpy(&nnDistancesSqr[i], &nearestNeighbors[i].distance, sizeof(T));
        std::memcpy(&nnIndexes[i], &nearestNeighbors[i].index, sizeof(int));
    }
}

template<typename T>
int CpuSearch<T>::getSpaceDim() {
    return spaceDim;
}

template<typename T>
CpuSearch<T>::~CpuSearch() {
    if(datasetAllocated)
        delete [] dataset;
    delete [] dists;
}



#endif //CUDA_SEARCH_HPP