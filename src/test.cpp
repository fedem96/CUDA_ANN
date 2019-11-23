//
// Created by iacopo on 21/11/19.
//
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <algorithm>


using namespace std;

struct SomeType {
    int key;
    int value;
    bool operator>(const SomeType& rhs) { return key > rhs.key; }
    bool operator>=(const SomeType& rhs) { return key >= rhs.key; }
    bool operator<(const SomeType& rhs) { return key < rhs.key; }
    bool operator<=(const SomeType& rhs) { return key <= rhs.key; }
};

int comparison_function(const SomeType& a, const SomeType& b ){
    return (a.key == b.key) ? 0 : ((a.key > b.key) ? 1 : -1);
}

int main() {
    vector<SomeType> v;
    int size = 10;
    for(int i=0; i<size; i++){
        SomeType s;
        s.key = rand() % size;
        s.value = i+1;
        v.push_back(s);
    }
    for(int i=0; i<size; i++) cout << "Key: " << v.at(i).key << " Value: " << v.at(i).value << endl;
    sort(v.begin(), v.end());
//	sort(v.begin(), v.end(), comparison_function);
    for(int i=0; i<size; i++) cout << "Sorted Key: " << v.at(i).key << " Value: " << v.at(i).value << endl;
}