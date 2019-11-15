#include <cassert>
/**
 * @brief from vector to string
 * @tparam T
 * @param vec: vector to convert
 * @return the string
 * Convert a vector into string, allowing a print of it
 */
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

/**
 * @brief Check results of a single query
 * @tparam T
 * @param groundTruth: true nearest neighbors (it's a vector of the indexes in the dataset)
 * @param indexes: indexes of nearest neighbors returned from the search method
 * @param distances: distances of nearest neighbors returned from the search method
 * @return true if given indexes and distances are coherent to the groundTruth
 *
 * Check result of a single query: result is coherent to groundTruth if indexes appear in the same order.
 * A different order is allowed only between elements at the same distance.
 */
template <typename T>
bool checkKNN(const std::vector<int> &groundTruth, const std::vector<int> &indexes, const std::vector<T> &distances){
    assert(indexes.size() == distances.size());
    int checkableElements = groundTruth.size() < indexes.size() ? groundTruth.size() : indexes.size();
    for(int i=0; i < checkableElements; i++){
        bool match = false;

        for(int j=i; j < checkableElements && distances[i] == distances[j]; j++)
            if(indexes[j] == groundTruth[i])
                match = true;

        if(!match)
            for(int j=i-1; j >= 0 && distances[i] == distances[j]; j--)
                if(indexes[j] == groundTruth[i])
                    match = true;
        if(!match) {
            std::cout << "assertion failed: i=" << i << ", indexes[i]=" << indexes[i] << ", groundTruth[i]=" << groundTruth[i] << std::endl;
            return false;
        }
    }
    //std::cout << "assert OK" << std::endl;
    return true;
}

/**
 * @param dataset
 * @param groundTruth
 * @param queries
 *
 * Some print to understand data
 */
// TODO cancellare questa funzione
void dataPrint(std::vector< std::vector<float> > &dataset, std::vector< std::vector<int> > &groundTruth, std::vector< std::vector<float> > &queries){

    std::cout << std::endl;
    std::cout << "base dataset size:\n\t" << dataset.size() << std::endl;
    std::cout << "groundtruth dataset size:\n\t" << groundTruth.size() << std::endl;
    std::cout << "query dataset size:\n\t" << queries.size() << std::endl;

    std::cout << std::endl;
    std::cout << "ELEMENT 0" << std::endl;
    std::cout << "\nelement 0 of vector base (" << dataset[0].size() << " elements):\n\t" << vecToStr<float>(dataset[0]) << std::endl;
    std::cout << "\nelement 0 of vector groundtruth (" << groundTruth[0].size() << " elements):\n\t" << vecToStr<int>(groundTruth[0]) << std::endl;
    std::cout << "\nelement 0 of vector query (" << queries[0].size() << " elements):\n\t" << vecToStr<float>(queries[0]) << std::endl;
    std::cout << std::endl;
    std::cout << "ELEMENT 13" << std::endl;
    std::cout << "\nelement 13 of vector base (" << dataset[13].size() << " elements):\n\t" << vecToStr<float>(dataset[13]) << std::endl;
    std::cout << "\nelement 13 of vector groundtruth (" << groundTruth[13].size() << " elements):\n\t" << vecToStr<int>(groundTruth[13]) << std::endl;
    std::cout << "\nelement 13 of vector query (" << queries[13].size() << " elements):\n\t" << vecToStr<float>(queries[13]) << std::endl;

    std::cout << std::endl;
    std::cout << "SIZES" << std::endl;
    std::cout << "\ngroundtruth sizeof: " << sizeof(groundTruth) << " bytes" << std::endl;
    std::cout << "groundtruth element sizeof: " << sizeof(groundTruth[13]) << " bytes" << std::endl;
}