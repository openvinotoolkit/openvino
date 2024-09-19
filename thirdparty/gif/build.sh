SOURCE_DIR=`dirname $0`
echo "SOURCE_DIR=${SOURCE_DIR}"
BUILD_DIR=`pwd`
echo "BUILD_DIR=${BUILD_DIR}"
rm -rf ${BUILD_DIR}/include
rm -rf ${BUILD_DIR}/lib
cp ${SOURCE_DIR}/giflib-5.1.9.tar.gz ./
tar zxf giflib-5.1.9.tar.gz
cd giflib-5.1.9
make -j8
mkdir -p ../include
mkdir -p ../lib
cp libgif.a ../lib
cp gif_lib.h ../include
