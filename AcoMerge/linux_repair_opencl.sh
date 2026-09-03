scl enable devtoolset-2 bash
yum install -y curl make openssl-devel git
cd /tmp
curl -L -O https://cmake.org/files/v3.4/cmake-3.4.3.tar.gz
tar -zxvf cmake-3.4.3.tar.gz
cd cmake-3.4.3
./bootstrap --prefix=/usr/local
gmake
make
make install
cd /tmp
hash -r 
git clone https://github.com/KhronosGroup/OpenCL-Headers.git
cd OpenCL-Headers
git checkout v2020.12.18
mkdir build && cd build
cmake ..
make install
cd /tmp
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
cd OpenCL-ICD-Loader
git checkout v2020.12.18
mkdir build && cd build
cmake ..
make install
ldconfig
cd /io
scl enable devtoolset-2 bash
