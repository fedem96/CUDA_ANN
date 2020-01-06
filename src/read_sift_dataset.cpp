#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <bitset>

using namespace std;

//#define DEBUG


template<class T>
T computeVectorMedian(const vector<T>& data) {
	vector<T> temp = data;
	sort(temp.begin(), temp.end(), greater<T>());
	T median = (temp[temp.size()/2-1]+temp[temp.size()/2])/2;
#ifdef DEBUG
	for(int i=0; i<temp.size(); i++)
		cout << temp[i] << " ";
	cout << endl << "Median: " << median << endl;
#endif		
	return median;
}

template<class T>
void computeVectorQuartile(const vector<T>& data, T& first, T&median, T& third) {
	vector<T> temp = data;
	sort(temp.begin(), temp.end(), greater<T>());
	median = (temp[temp.size()/2-1]+temp[temp.size()/2])/2;
	first = (temp[temp.size()/4-1]+temp[temp.size()/4])/2;
	third = (temp[temp.size()*3/4-1]+temp[temp.size()*3/4])/2;
#ifdef DEBUG
	for(int i=0; i<temp.size(); i++)
		cout << temp[i] << " ";
	cout << endl << "Median: " << median << " - First: " << first << " - Third: " << third << endl;
#endif		
}


template<class T, int N>
bitset<N> binarizeVector(const vector<T>& data, T median) {
	bitset<N> result;
	for(int i=0; i < data.size(); i++) {
		if( data[i]>median )
			result[i]=1;
	}
	return result;
}

template<class T, class U>
bool readVecsFile(string fileName, vector< vector<U> >& vectors, bool verbose = true) {
	ifstream file(fileName.c_str(), ios::in | ios::binary | ios::ate);
	ifstream::pos_type fileSize;
	if( file.is_open() ) {
		if( verbose )
			cout << "Opening file: " << fileName << endl;
		fileSize = file.tellg();
		file.seekg(0, ios::beg);
		int vectorDim;
		file.read(reinterpret_cast<char*>(&vectorDim), 4);
		int vectorSize = 1*sizeof(int) + vectorDim*sizeof(T);
		int numVectors = ( fileSize / vectorSize );
		
		if( verbose ) {
			cout << "- File size: " << fileSize << endl;
			cout << "- Vector dimension: " << vectorDim << endl;
			cout << "- Expected number of vectors: " << numVectors << endl;
		}
			
		vectors.reserve( numVectors+vectors.size() );
		file.seekg(0, ios::beg);
		while( !file.eof() ) {
			int rawVectorDim;
			if( !file.read(reinterpret_cast<char*>(&rawVectorDim), sizeof(int)) ) {
				if( file.eof() )
					break;
				else {
					if( verbose )
						cerr << "ERROR: Fail to read vector dimension: " << fileName << endl;
					break;
				}
			}
			if( rawVectorDim != vectorDim) {
				if( verbose )
					cerr << "ERROR: Vector size changed: " << rawVectorDim << " (" << vectorDim << ")" << endl;
			}
			int rawVectorSize = vectorSize-sizeof(int);
			T* rawVectorData = new T[rawVectorSize];
			if( !file.read(reinterpret_cast<char*>(rawVectorData), rawVectorSize) ) {
				delete[] rawVectorData;
				if( file.eof() )
					break;
				else {
					if( verbose )
			       		cerr << "ERROR: Fail to read vector data: " << fileName << endl;
					break;
				}
			}
			vector<U> aVector;
			aVector.reserve( vectorDim );
			for( int i=0; i<rawVectorDim; i++ ) {
				aVector.push_back( static_cast<U>(rawVectorData[i]) );
			}
			delete[] rawVectorData;
			vectors.push_back( aVector );
		}
		file.close();
		if( vectors.size() != numVectors ) {
			cerr << "ERROR: read " << vectors.size() << " instead of " << numVectors << endl;
		} else {
			if( verbose )
				cout << "Read " << vectors.size() << " vectors" << endl;
		}
#ifdef DEBUG
		vector<T> aVec = (*vectors.begin());
		for(int i=0; i<aVec.size(); i++)
			cout << aVec[i] << " ";
		cout << endl;
		T first, third, median;
		median=computeVectorMedian(aVec);
		computeVectorQuartile(aVec, first, median, third);
		bitset<128> binarized = binarizeVector<T, 128>(aVec, median);
		for(int i=0; i<binarized.size(); i++)
			cout << binarized[i] << " ";
		cout << endl;	
#endif
	} else {
		if( verbose )
			cerr << "ERROR: Can not open file: " << fileName << endl;
		return false;
	}
	return true;
}


/*int main(int argc, char *argv[]) {
	string fileName;
	if( argc >=2 )
		fileName = argv[1];
	else {
		cerr << "ERROR: Missing file name" << endl;		return -1;
	}
	
	if( fileName.substr(fileName.length()-6,fileName.length())==".fvecs" ) {
		vector< vector<float> > fvecs;
		bool result = readVecsFile<float,float>(fileName, fvecs, true);
	}
	if( fileName.substr(fileName.length()-6,fileName.length())==".ivecs" ) {
		vector< vector<int> > ivecs;
		bool result = readVecsFile<int,int>(fileName, ivecs, true);
	}
}
*/

