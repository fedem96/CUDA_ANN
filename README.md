# CUDA_Nearest_Neighbors
![alt text](https://github.com/fedem96/Nearest_Neighbors/blob/master/img/k-nn.png)

The aim of this project is to measure the speedup obtained for the algorithm K Nearest Neighbors when implemented in OpenMP/CUDA with respect to the C++ sequential version.

## System configuration
+ Intel Core i7-8750H @ 2.20Ghz (up to 4.10Ghz with Turbo Boost)
+ 16 GiB RAM
+ NVIDIA GeForce GTX 1050 Ti (Notebook)
+ CUDA 10.0

## Datasets
Sift and siftsmall datasets can be found in http://corpus-texmex.irisa.fr, where they are called ANN_SIFT1M and ANN_SIFT10K respectively.

## Experiments
### Download SIFT dataset
* `wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`
* `mkdir data`
* `tar -xf sift.tar.gz -C data`

### Download this repo
* `git clone https://github.com/fedem96/Nearest_Neighbors.git`

### Run the experiments
* `mkdir experiments`
* `cd Nearest_Neighbors`
* `nvcc src/main.cu -Xcompiler -fopenmp -o run`
* `./run -d ../data -e ../experiments`
