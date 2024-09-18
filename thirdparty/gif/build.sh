tar zxvf giflib-5.1.9.tar.gz
cd giflib-5.1.9
make -j8
rm -rf ../include
rm -rf ../lib
mkdir -p ../include
mkdir -p ../lib
cp libgif.a ../lib
cp gif_lib.h ../include
cd ../
rm -rf giflib-5.1.9
