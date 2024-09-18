tar zxvf libpng-1.6.37.tar.gz 
cd libpng-1.6.37
mkdir build
cd build
cmake ../ -DPNG_SHARED=OFF -DPNG_TESTS=OFF -DCMAKE_INSTALL_PREFIX=./_install
make -j8
make install
rm -rf ../../include
rm -rf ../../lib
cp -r _install/include ../../
cp -r _install/lib ../../
cd ../../
rm -rf libpng-1.6.37

