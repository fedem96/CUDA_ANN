#ifndef CUDA_SEARCH_HPP
#define CUDA_SEARCH_HPP

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include <cstring>
//#include <thrust/sort.h>


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

};

// TODO the contructor must specify the number of cores for the openMP version
template<typename T>
CpuSearch<T>::CpuSearch(T *dataset, int datasetSize, int spaceDim, int numCores) : dataset(dataset), datasetSize(datasetSize), spaceDim(spaceDim), datasetAllocated(false), numCores(numCores), nnAllIndexes(datasetSize), nnAllDistancesSqr(datasetSize), nearestNeighbors(datasetSize) {
    assert(datasetSize > 0);
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
}

template<typename T>
T compare (const void * a, const void * b)
{
    return ( *(T*)a - *(T*)b );
}


//template<typename T>
//int comparator(const Neighbor<T> &a, const Neighbor<T> &b){
//    return (a.distance == b.distance) ? 0 : ((a.distance > b.distance) ? 1 : -1);
//}

template<typename T>
T comparator(const void * a, const void * b){
    return ((Neighbor<T> *)a)->distance - ((Neighbor<T> *)b)->distance;
}

template<typename T>
void CpuSearch<T>::search(T* query, std::vector<int> &nnIndexes, std::vector<T> &nnDistancesSqr, const int &numResults){
    // TODO finire di implementare la ricerca, versione cpu sequenziale / OpenMP
    //int tid;
    omp_set_num_threads(numCores);
#pragma omp parallel for //private(numCores)
    //#pragma for schedule(auto)
    //#pragma for schedule(dynamic)
    for(int i=0; i < datasetSize ; i++) {
        T dist = 0;
        //#pragma omp critical
        for (int j = 0; j < spaceDim; j++) {
            const T diff = query[j] - dataset[i * spaceDim + j];
            dist = dist + (diff * diff);
        }

        //tid = omp_get_thread_num();
        // printf(" thread id = %d - i = %d \n", tid , i);
        // dist is the square of the distance: for efficiency I use it instead of distance to find neighbors
//        nnAllDistancesSqr[i] = dist;
//        nnAllIndexes[i] = i;
        nearestNeighbors[i].distance = dist;
        nearestNeighbors[i].index = i;
    }

    // sorting whit thrust is expensive why?
    // sort by increasing distance

    //thrust::sort_by_key(&nnAllDistancesSqr[0], &nnAllDistancesSqr[0]+this->datasetSize, &nnAllIndexes[0]);
    //std::sort(&nnAllDistancesSqr[0], &nnAllDistancesSqr[0]+this->datasetSize);
    //qsort (&nnAllDistancesSqr[0], this->datasetSize, sizeof(T), compare);

    //std::sort(&nearestNeighbors[0], &nearestNeighbors[0]+this->datasetSize);
    //qsort (&nearestNeighbors[0], this->datasetSize, sizeof(Neighbor<T>), comparator);

    // copy distances and indexes of nearest neighbors
//    std::memcpy(&nnDistancesSqr[0], &nnAllDistancesSqr[0], sizeof(T)  * numResults);
//    std::memcpy(&nnIndexes[0], &nnAllIndexes[0], sizeof(int) * numResults);

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
}



#endif //CUDA_SEARCH_HPP