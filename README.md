# quantizeLLM

ig @zingraphics

quantizing LLM
python -m torch.distributed.launch  to launch in parallel


BUILDING PYTORCH
To build Caffe2 on Ubuntu, you need to install the following dependencies:
PyTorch build process builds Caffe2 as well. Caffe2 is a deep learning framework that is integrated into PyTorch as a sub-library.
When you build PyTorch from source, the build process compiles the Caffe2 C++ code, generates the Python bindings, and includes the Caffe2 library in the PyTorch package.
1.) Basic build tools: sudo apt-get install build-essential
2.) Git: sudo apt-get install git
3.) CMake: sudo apt-get install cmake
4.) BLAS: sudo apt-get install libatlas-base-dev
5.) OpenCV: sudo apt-get install libopencv-dev
6.) Python: sudo apt-get install python3-dev python3-pip
7.) NumPy: pip3 install numpy
8.) Protocol Buffers: sudo apt-get install libprotobuf-dev protobuf-compiler
9.) LevelDB: sudo apt-get install libleveldb-dev
10.) LMDB: sudo apt-get install liblmdb-dev
11.) SNAPPY: sudo apt-get install libsnappy-dev
12.) CUDA (if you plan to build with GPU support): Follow the instructions provided by NVIDIA to install CUDA on your system.

To install these dependencies, you can run the following command:
sudo apt-get install build-essential git cmake libatlas-base-dev libopencv-dev python3-dev python3-pip libprotobuf-dev protobuf-compiler libleveldb-dev liblmdb-dev libsnappy-dev
maxrregcount value specified 96, will be ignored ptxas warning : Too big
we change this to 43 older than k10 and to get a successfull buil currently testing values

older thank k10 96 
apt install python3.8-venv
python3 -m venv pytorch-env
source pytorch-env/bin/activate
cd pytorch
mkdir build
cd build
cmake ..
cd ..
python3 setup.py install
export LD_LIBRARY_PATH=/opt/intel/lib/intel64:$LD_LIBRARY_PATH
export CMAKE_INCLUDE_PATH=/opt/intel/mkl/include:$CMAKE_INCLUDE_PATH
export CMAKE_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$CMAKE_LIBRARY_PATH

you also may want to build tensorflow from source to have it tailored to your specific system requirements leading to better performance and less errors or complications depending on the 

For the language model:
QUANTIZING THE MODEL 
SAVE QUANTIZE MODEL
LOAD



