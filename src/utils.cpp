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
        bool cond = (indexes[i] == groundtruth[i]) || (i>0 && distances[i-1] == distances[i]) || (i<indexes.size() && distances[i] == distances[i+1]);
        if(! cond)
            std::cout << "assertion failed: i=" << i << ", indexes[i]=" << indexes[i] << ", groundtruth[i]=" << groundtruth[i] << std::endl;
        assert(cond);
    }
    //std::cout << "assert OK" << std::endl;
    return true;
}