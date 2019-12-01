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

template<typename Iterator>
Iterator merge(Iterator ii1, Iterator end1, Iterator ii2, Iterator end2, Iterator oi, Iterator endO);

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
    CpuSearch(const std::vector< std::vector<T> > &dataset, int numThreads=0);
    void search(T* query, std::vector<int> &nnIndexes, std::vector<T> &nnDistancesSqr, const int &numResults) override;
    int getSpaceDim() override;
    virtual ~CpuSearch();
private:
    T* dataset;
    int datasetSize;
    int spaceDim;
    bool datasetAllocated;
    int numThreads;

    std::vector<Neighbor<T> > nearestNeighbors;

    T* dists;

    int chunkSize;
    int remainder;

};

// TODO the contructor must specify the number of cores for the openMP version
template<typename T>
CpuSearch<T>::CpuSearch(T *dataset, int datasetSize, int spaceDim, int numCores) : dataset(dataset), datasetSize(datasetSize), spaceDim(spaceDim), datasetAllocated(false), numThreads(numCores), nearestNeighbors(datasetSize) {
    assert(datasetSize > 0);
    dists = new T[spaceDim];


    chunkSize = datasetSize / numThreads;
    remainder = datasetSize - (chunkSize * numThreads);
}

template<typename T>
CpuSearch<T>::CpuSearch(const std::vector< std::vector<T> > &dataset_vv, int numThreads) : datasetSize(dataset_vv.size()), numThreads(numThreads), nearestNeighbors(dataset_vv.size())
{
    assert(datasetSize > 0);
    spaceDim = dataset_vv.at(0).size();
    dists = new T[spaceDim];
    dataset = new T[datasetSize * spaceDim];
    datasetAllocated = true;
    for(int i=0; i < datasetSize; i++){
        std::memcpy(dataset + (i*spaceDim), &dataset_vv[i][0], sizeof(T) * spaceDim);
    }

    chunkSize = datasetSize / numThreads;
    remainder = datasetSize - (chunkSize * numThreads);
}

template<typename T>
T comparator(const void * a, const void * b){
    return ((Neighbor<T> *)a)->distance - ((Neighbor<T> *)b)->distance;
}

template<typename T>
void CpuSearch<T>::search(T* query, std::vector<int> &nnIndexes, std::vector<T> &nnDistancesSqr, const int &numResults){

    omp_set_num_threads(numThreads);
    #pragma omp parallel
    {
        // calculate distances
        #pragma omp for
        for (int i = 0; i < datasetSize; i++) {
            T dist = 0;
            T *dataset_tmp = dataset + (i * spaceDim);
            for (int j = 0; j < spaceDim; j++) {
                const T diff = query[j] - dataset_tmp[j];
                dist += diff * diff;
            }

            // dist is the square of the distance: for efficiency I use it instead of distance to find neighbors
            nearestNeighbors[i].distance = dist;
            nearestNeighbors[i].index = i;
        }

        // sort by increasing distance
        #pragma omp for
        for (int i = 0; i < numThreads; i++) {  // each thread sorts its chunk
            qsort(&nearestNeighbors[i * chunkSize], chunkSize + (i + 1 == numThreads ? remainder : 0), sizeof(Neighbor<T>), comparator);
        }
    }

    // in single-thread, no merge is required
    if(numThreads != 1) {
        // sorted chunk needs to be merged
        Neighbor<T> *tmp = new Neighbor<T>[numResults];
        for (int i = 1; i < numThreads; i++) {
            std::memcpy(tmp, &nearestNeighbors[0], sizeof(Neighbor<T>) * numResults);
            merge(tmp, tmp + numResults, &nearestNeighbors[i * chunkSize], &nearestNeighbors[i * chunkSize + numResults], &nearestNeighbors[0], &nearestNeighbors[numResults]);
        }
        delete [] tmp;
        // at the end of merge, nearest neighbors are the first elements of the array nearestNeighbors
    }

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


template<typename Iterator>
Iterator merge(Iterator ii1, Iterator end1, Iterator ii2, Iterator end2, Iterator oi, Iterator endO){
    while((ii1 < end1 || ii2 < end2) && oi < endO)
        if(ii1 < end1 && *ii1 <= *ii2)
            *(oi++) = *(ii1++);
        else
            *(oi++) = *(ii2++);
    return oi;
}


#endif //CUDA_SEARCH_HPP