tar zxvf libjpeg.tgz
cd libjpeg
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=./_install
make -j8
make install
rm -rf ../../include
rm -rf ../../lib
cp -r _install/include ../../
cp -r _install/lib ../../
cd ../../
rm -rf libjpeg

