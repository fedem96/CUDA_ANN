#include <cassert>

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

template <typename T>
bool checkKNN(const std::vector<int> &groundtruth, const std::vector<size_t> &indexes, const std::vector<T> &distances){
    assert(indexes.size() == distances.size());
    int checkableElements = groundtruth.size() < indexes.size() ? groundtruth.size() : indexes.size();
    for(size_t i=0; i < checkableElements; i++){
        bool match = false;
        for(size_t j=i; j < checkableElements && distances[i] == distances[j]; j++)
            if(indexes[j] == groundtruth[i])
                match = true;

        if(!match)
            for(size_t j=i; j >= 0 && distances[i] == distances[j]; j--)
                if(indexes[j] == groundtruth[i])
                    match = true;
        if(!match)
            std::cout << "assertion failed: i=" << i << ", indexes[i]=" << indexes[i] << ", groundtruth[i]=" << groundtruth[i] << std::endl;
        assert(match);
    }
    //std::cout << "assert OK" << std::endl;
    return true;
}

void dataPrint(std::vector< std::vector<float> > &baseVec, std::vector< std::vector<int> > &grthVec, std::vector< std::vector<float> > &queryVec){

    std::cout << std::endl;
    std::cout << "base dataset size:\n\t" << baseVec.size() << std::endl;
    std::cout << "groundtruth dataset size:\n\t" << grthVec.size() << std::endl;
    std::cout << "query dataset size:\n\t" << queryVec.size() << std::endl;

    std::cout << std::endl;
    std::cout << "ELEMENT 0" << std::endl;
    std::cout << "\nelement 0 of vector base (" << baseVec[0].size() << " elements):\n\t" << vecToStr<float>(baseVec[0]) << std::endl;
    std::cout << "\nelement 0 of vector groundtruth (" << grthVec[0].size() << " elements):\n\t" << vecToStr<int>(grthVec[0]) << std::endl;
    std::cout << "\nelement 0 of vector query (" << queryVec[0].size() << " elements):\n\t" << vecToStr<float>(queryVec[0]) << std::endl;
    std::cout << std::endl;
    std::cout << "ELEMENT 13" << std::endl;
    std::cout << "\nelement 13 of vector base (" << baseVec[13].size() << " elements):\n\t" << vecToStr<float>(baseVec[13]) << std::endl;
    std::cout << "\nelement 13 of vector groundtruth (" << grthVec[13].size() << " elements):\n\t" << vecToStr<int>(grthVec[13]) << std::endl;
    std::cout << "\nelement 13 of vector query (" << queryVec[13].size() << " elements):\n\t" << vecToStr<float>(queryVec[13]) << std::endl;

    std::cout << std::endl;
    std::cout << "SIZES" << std::endl;
    std::cout << "\ngroundtruth sizeof: " << sizeof(grthVec) << " bytes" << std::endl;
    std::cout << "groundtruth element sizeof: " << sizeof(grthVec[13]) << " bytes" << std::endl;
}