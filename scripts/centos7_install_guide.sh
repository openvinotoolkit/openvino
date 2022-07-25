opyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

############################################################
# Function                                                 #
############################################################
Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo
   echo "Syntax: scriptTemplate [-h|g|p|c|m|b]"
   echo "options:"
   echo "h     print this Help."
   echo "g     gcc version(default 8)"
   echo "p     python version(default 7)"
   echo "c     cmake dir(default "~")"
   echo "m     model name(default resnet-50-pytorch)" 
   echo "b     use benchmark_app c++(default false)"
}

usage() 
{ echo "Usage: [-g <7 or 8>] [-p <6, 7, 8 or 9>] [-c <cmake dir>] [-m <evaluation model>] \
 [-b benchmark_app c++ <true or false>] " \
 [-h help] 1>&2; exit 1; }

CppBenchmarkFunc()
{
cd $ovDir/openvino/$installDir/samples/cpp/

./build_samples.sh -b .
benchmark_appPath=$ovDir/openvino/$installDir/samples/cpp/intel64/Release/benchmark_app
if command -v $benchmark_appPath; then
  echo "$benchmark_appPath exists"
  $benchmark_appPath -m ~/ov_models/public/$model/FP32/$model.xml -d CPU 
else
  echo "not find $benchmark_appPath"
  exit 1
 fi
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

echo "############################################################"
# get the absolute path from script's dir
ovDir=$(cd `dirname $0` && cd ../.. && pwd)
cd $ovDir/openvino
# check OS version
if [ -f /etc/redhat-release ]; then
  echo "Demo to install and evaluate OV on centos7.x"
else 
  echo "This script is only for centos7"
  exit 1
fi

echo "current openvino(tag): $(git describe --tags)"
# Set variables
gccSet=7
pySet=7
cmakeDir=~
model=resnet-50-pytorch
benchmarkCpp=false

############################################################
# Process the input options. Add options as needed.        #
############################################################

# Get the options
while getopts "g:p:c:m:b:h" option; do
    case "${option}" in
        g) gccSet=$OPTARG;;
        p) pySet=$OPTARG;;
        c) cmakeDir=$OPTARG;;
        m) model=$OPTARG;;
        b) benchmarkCpp=$OPTARG;;
        h) Help 
        exit;; # display Help
        \?) usage;;      
    esac
done

# if no arg, show the usage
if [ $OPTIND -eq 1 ]; then
    usage
    exit 1
fi 
# check gccSet pySet
if [ $gccSet -eq 7 -o  $gccSet -eq 8 ]; then
  echo "set gcc version: $gccSet";
else    
  usage
  exit 1
fi 
# check py
if [ $pySet -eq 6 -o $pySet -eq 7 -o $pySet -eq 8 -o  $pySet -eq 9 ]; then
  echo "set python version: 3.$pySet";
else    
  usage
  exit 1
fi 
echo "cmake Dir: $cmakeDir"
echo "evaluation model: $model"
echo "use benchmark_app c++: $benchmarkCpp"

############################################################
#     0.system dependency and environment                  #
############################################################
echo "############################################################"
echo ">>> 0.system dependency and environment"
echo "install yum, gcc, dnf, centos-release-scl, git" 
sudo yum update
sudo yum -y install gcc dnf centos-release-scl git
echo ">>> install anaconda3"

if [ ! -d ~/anaconda3 ]; then
  # Download anaconda3 for python env
  wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
  sudo chmod +x Anaconda3-2020.11-Linux-x86_64.sh
  # silent installation
  ./Anaconda3-2020.11-Linux-x86_64.sh -b
  ~/anaconda3/condabin/conda init
  source ~/.bashrc
  conda config --set auto_activate_base false
  conda deactivate # base env
else 
  echo "anaconda3 exists"
fi

############################################################
#     1.Download CMake                                     #
############################################################
echo "############################################################"
echo ">>> 1.Download CMake 3.18.4" 

cmakeVersion=cmake-3.18.4-Linux-x86_64

if [ ! -d $cmakeDir/$cmakeVersion ]; then
   echo "$cmakeVersion not exit, now download"
   wget https://cmake.org/files/v3.18/$cmakeVersion.tar.gz \
   --directory-prefix $cmakeDir
   tar -xvf $cmakeDir/$cmakeVersion.tar.gz -C $cmakeDir
fi
echo "$cmakeVersion exists"
export PATH=$cmakeDir/$cmakeVersion/bin:$PATH

############################################################
#     2.Install devtoolset and setup environment           #
############################################################
echo "############################################################"
echo ">>> 2.Install devtoolset-$gccSet and setup environment"

if [ ! -d /opt/rh/devtoolset-$gccSet ]; then
  echo "devtoolset-$gccSet not exists, now download"
  sudo yum -y install devtoolset-$gccSet
else 
  echo "devtoolset-$gccSet exists"
fi

############################################################
#     2.1 Enable devtoolset                                #
############################################################

echo "############################################################"
echo ">>> 2.1 Enable devtoolset-$gccSet"
# instead $scl enable devtoolset-8 bash, no need to exit
source /opt/rh/devtoolset-$gccSet/enable
gccVersion=$(gcc --version | grep gcc | awk '{print $3}')
echo "use devtoolset, now gcc version is $gccVersion."

echo "############################################################"
echo ">>> 2.2 use anaconda3 to create py3$pySet env"
pyPath=~/anaconda3/envs/py3$pySet
if [ ! -d $pyPath ]; then
  echo "$pyPath not exists, now conda create"
  conda create -n py3$pySet python=3.$pySet
else
  echo "$pyPath exists."
fi
echo "############################################################"
echo ">>> 2.3 Activate py3$pySet env"
source ~/anaconda3/bin/activate py3$pySet
pyVersion=$(python --version 2>&1| awk '{print $2}')

if [ ${pyVersion:2:1} -eq $pySet ]; then
  echo "conda activate correctly, now python version is $pyVersion."
else    
  echo "conda error, now python version is $pyVersion."
  exit 1
fi 

############################################################
#     3. Build OV with cmake                               #
############################################################
echo "############################################################"
echo ">>> 3. Build OV with cmake"
echo ">>> 3.1 openvino git submodule update"
cd $ovDir/openvino
git submodule update --init --recursive

echo "############################################################"
echo ">>> 3.2 pip install python dependency"
pip install -U pip wheel setuptools cython patchelf

echo "############################################################"
echo ">>> 3.3 cmake to build OV"
# different dir to cmake, e.g. "build_gcc8_py39", "install_gcc8_py38"
buildDir=build_gcc${gccSet}"_py3"${pySet}
installDir=install_gcc${gccSet}"_py3"${pySet}

if [ -d $buildDir ]; then 
  echo "delete $buildDir"
  rm -rf $buildDir
fi
if [ -d $installDir ]; then 
  echo "delete $installDir"
  rm -rf $installDir
fi
if [ -d temp ]; then 
  echo "delete temp"
  rm -rf temp
fi
mkdir -p $buildDir && mkdir -p $installDir && cd $buildDir

pathDPYTHON_EXECUTABLE=$pyPath/bin/python
pathDPYTHON_LIBRARY=$(find $pyPath/lib -maxdepth 1 -name libpython3.$pySet*.so)
pathDPYTHON_INCLUDE_DIR=$(find $pyPath/include -maxdepth 1 -name python3.$pySet*)

if [ ! -f $ovDir/openvino/$installDir/tools/openvino-2022*.whl ] ; then
  echo "whls not exist"
fi
echo "cmake now"
# will download prebuild TBB instead of using system's 
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON \
-DCMAKE_INSTALL_PREFIX=../$installDir -DENABLE_SYSTEM_TBB=OFF -DENABLE_OPENCV=OFF \
-DENABLE_INTEL_GNA=OFF -DENABLE_INTEL_MYRIAD_COMMON=OFF -DTREAT_WARNING_AS_ERROR=OFF \
-DPYTHON_EXECUTABLE=$pathDPYTHON_EXECUTABLE \
-DPYTHON_LIBRARY=$pathDPYTHON_LIBRARY \
-DPYTHON_INCLUDE_DIR=$pathDPYTHON_INCLUDE_DIR  \
..

make --jobs=$(nproc --all)
make install

if [ ! -f $ovDir/openvino/$installDir/tools/openvino-2022*.whl ] ; then
  echo "whls not exist, cmake fails"
  exit 1
else
  cd $ovDir/openvino/$installDir/tools/
  echo openvino-2022*.whl exits
  echo openvino_dev-2022*.whl exits
fi


############################################################
#     4.Install python wheel                               #
############################################################
echo "############################################################"
echo ">>> 4.Install python wheel"

if [ ! $(command -v mo) ] ; then
  echo "now mo not exists"
fi 

cd $ovDir/openvino/$installDir/tools
echo install openvino-2022*.whl openvino_dev-2022*.whl
pip install --force-reinstall openvino-2022*.whl openvino_dev-2022*.whl  

if [ ! $(command -v mo) ] ; then
  echo "mo STILL not exists, whls installation fail" && exit 1
fi  
############################################################
#     5.Model evaluation                                   #
############################################################
echo "############################################################"
echo ">>> 5.$model evaluation with benchmark_app"

############################################################
#     5.1 prepare dependency,download and convert model    #
############################################################
mkdir -p ~/ov_models

if [ ! -d ~/ov_models/public/$model ]; then
  echo "$model not exists"
fi
echo "now install onnx, pytorch, omz_downloader and omz_converter"
pip install protobuf==3.16.0 
pip install onnx==1.11.0 # python3.6 not support ONNX 1.12
pip install openvino-dev[pytorch] # install ONNX's dependency
omz_downloader --name $model -o ~/ov_models/
omz_converter --name $model -o ~/ov_models/ -d ~/ov_models/

############################################################
#     5.2 run benchmark_app                                #
############################################################
echo "############################################################"

if ! $benchmarkCpp; then
  echo "use default benchmark_app python version"
  benchmark_app -m ~/ov_models/public/$model/FP32/$model.xml -d CPU 
else echo "use benchmark_app c++ version"
  CppBenchmarkFunc
fi

echo "############################################################"
echo "Congratulation! centos7-install-guide is finished."
echo "############################################################"
echo "Here is an OV usage example on centos7:"
echo "source /opt/rh/devtoolset-$gccSet/enable"
echo "export PATH=\"~/anaconda3/bin:\$PATH\""
echo "conda activate py3$pySet"
echo "benchmark_app -m ~/ov_models/public/$model/FP32/$model.xml -d CPU"
echo "############################################################"
conda deactivate && cd $ovDir/openvino 
exit 1


