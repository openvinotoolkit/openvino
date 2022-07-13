#!/bin/sh

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

############################################################
# Function                                                 #
############################################################
Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo
   echo "Syntax: scriptTemplate [-h|g|p|c|m]"
   echo "options:"
   echo "h     Print this Help."
   echo "g     gcc version(default 8)"
   echo "p     python version(default 7)"
   echo "c     cmake file path(default "~/")"
   echo "m     model name(default resnet-50-pytorch)" 
   echo "b     Use benchmark_app c++(default false)"
}

usage() 
{ echo "Usage: [-g <7 or 8>] [-p <6, 7, 8 or 9>] [-c <cmake file path>] [-m <evaluation model>] [-b <true or false>]" 1>&2; exit 1; }

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

# added chmod 755 script.sh

# check os version
if [ -f /etc/lsb-release ]; then
  echo Ubuntu
elif [ -f /etc/redhat-release ]; then
  echo "Demo to install and evaluate OV on centos7.x"
elif [ -f /etc/os-release ] && grep -q "raspbian" /etc/os-release; then
  echo Raspbian

fi
# Set variables

gccSet=8
pySet=7
cmakePath=~
model=resnet-50-pytorch
benchmarkCpp=false

############################################################
# Process the input options. Add options as needed.        #
############################################################

# Get the options
while getopts "g:p:c:m:h:b:" option; do
    case "${option}" in
        g) gccSet=$OPTARG;;
        p) pySet=$OPTARG;;
        c) cmakePath=$OPTARG;;
        #o) ovPath=$OPTARG;;
        m) model=$OPTARG;;
        b) benchmarkCpp=$OPTARG;;
        h) Help 
        exit;; # display Help
        *) usage;;         
    esac
done

if [ $OPTIND -eq 1 ]; then
    usage
    exit 1
fi 

echo "set gcc version: $gccSet";
echo "set python version: 3.$pySet";

############################################################
#     0.system dependency and environment                  #
############################################################

echo ">>> 0.system dependency and environment"

<<comment
sudo -i
echo “proxy=http://child-prc.intel.com:913” >> /etc/yum.conf
exit
sudo yum update
sudo yum install gcc dnf centos-release-scl git
comment
echo proxy yum gcc dnf centos-release-scl git exist

# download anaconda3 for python env
if [ -d ~/anaconda3 ]; then
  echo anaconda3 exists
else 
  wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
  sudo chmod +x Anaconda3-2020.11-Linux-x86_64.sh
  ./Anaconda3-2020.11-Linux-x86_64.sh
fi

############################################################
#     1.Download CMake 3.18.4                              #
############################################################
echo "############################################################"
echo ">>> 1.Download CMake 3.18.4" 
echo "############################################################"
DIC=cmake-3.18.4-Linux-x86_64

if [ -d $cmakePath/$DIC ]; then
    echo "$DIC exists."
else
   echo "$DIC not exit, now download"
   wget https://cmake.org/files/v3.18/cmake-3.18.4-Linux-x86_64.tar.gz \
   --directory-prefix $cmakePath
   tar -xvf $cmakePath/cmake-3.18.4-Linux-x86_64.tar.gz -C $cmakePath
   export PATH=$cmakePath/cmake-3.18.4-Linux-x86_64/bin:$PATH
fi

############################################################
#     2.Install devtoolset and setup environment           #
############################################################

echo ">>> 2.Install devtoolset-$gccSet and setup environment"

if [ -d /opt/rh/devtoolset-$gccSet ]; then
  echo "devtoolset-$gccSet exists"
else 
  echo "devtoolset-$gccSet not exists, now download"
  sudo yum -y install devtoolset-$gccSet
fi

############################################################
#     2.1 Enable devtoolset                                #
############################################################

# Countermeasures for the message on the right: "MANPATH: unbound variable"
# now comment, for it causes conda activate error 
# MANPATH="" 

# instead $scl enable devtoolset-8 bash, no need to exit
source /opt/rh/devtoolset-$gccSet/enable
gccVersion=$(gcc --version | grep gcc | awk '{print $3}')
echo "use devtoolset, now gcc version is $gccVersion."
# conda create python env
pyPath=~/anaconda3/envs/py3$pySet
if [ -d $pyPath ]; then
  echo "$pyPath exists."
  conda activate py3$pySet
  pyVersion=$(python --version 2>&1| awk '{print $2}')
  echo "now python version is $pyVersion."
else
  echo "$pyPath not exists, now conda create"
  conda create -n py3$pySet python=3.$pySet
fi

# check python path before cmake
pathDPYTHON_EXECUTABLE=$pyPath/bin/python
#echo $pathDPYTHON_EXECUTABLE
pathDPYTHON_LIBRARY=$(find $pyPath/lib -maxdepth 1 -name libpython3.$pySet*.so)
#echo $pathDPYTHON_LIBRARY
pathDPYTHON_INCLUDE_DIR=$(find $pyPath/include -maxdepth 1 -name python3.$pySet*)
#echo $pathDPYTHON_INCLUDE_DIR

############################################################
#     3. Build OV with cmake                               #
############################################################

echo ">>> 3. Build OV with cmake"

if [ -d $ovPath/openvino ]; then
  echo OV exists
  git submodule update --init --recursive
  pip install -U pip wheel setuptools cython patchelf
else 
  #cd 
  #git clone https://github.com/openvinotoolkit/openvino -b 2022.1.0
  #cd openvino
  git submodule update --init --recursive
  pip install -U pip wheel setuptools cython patchelf
fi

cd ../..
ovPath=${PWD}

cd $ovPath/openvino
# different dic to cmake, e.g. "build_gcc8_py39", "install_gcc8_py38"
buildDic=build_gcc${gccSet}"_py3"${pySet}
installDic=install_gcc${gccSet}"_py3"${pySet}
mkdir -p $buildDic && mkdir -p $installDic && cd $buildDic

if [ -f $ovPath/openvino/$installDic/tools/openvino-2022.1.0-000-cp3${pySet}* ]; then
  echo "whls exist and whls were already installed "
  
  # check mo and benchmark_app 
 if ! command -v mo; then
  echo "mo not exists"
  echo ">>> 4.Install python wheel"
  cd $ovPath/openvino/$installDic/tools && pip install openvino-2022* openvino_dev* 
 fi
 
else 
  export PATH=$cmakePath/cmake-3.18.4-Linux-x86_64/bin:$PATH
  echo "whls not exist, now cmake"
  # not need OpenCV and TBB
  cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON \
  -DCMAKE_INSTALL_PREFIX=../$installDic -DENABLE_SYSTEM_TBB=OFF -DENABLE_OPENCV=OFF \
  -DENABLE_INTEL_GNA=OFF -DENABLE_INTEL_MYRIAD_COMMON=OFF -DTREAT_WARNING_AS_ERROR=OFF \
  -DPYTHON_EXECUTABLE=$pathDPYTHON_EXECUTABLE \
  -DPYTHON_LIBRARY=$pathDPYTHON_LIBRARY \
  -DPYTHON_INCLUDE_DIR=$pathDPYTHON_INCLUDE_DIR  \
  ..

  make --jobs=$(nproc --all)
  make install

############################################################
#     4.Install python wheel                               #
############################################################

  echo ">>> 4.Install python wheel"
  
  cd $ovPath/openvino/$installDic/tools
  pip install openvino-2022* openvino_dev*  
fi

############################################################
#     5.Model evaluation                                   #
############################################################

echo ">>> 5.$model evaluation with benchmark_app"

mkdir -p ~/ov_models

if [ -d ~/ov_models/public/$model ]; then
  echo "$model exists, evaluate with benchmark_app directly"
else 
############################################################
#     5.1 prepare dependency,download and convert model    #
############################################################

  echo "$model not exists, now install onnx, pytorch, omz_downloader and omz_converter"
  pip install onnx==1.11.0 # python3.6 not support ONNX 1.12
  pip install openvino-dev[pytorch] # Pytorch will install ONNX's dependency
  omz_downloader --name $model -o ~/ov_models/
  omz_converter --name $model -o ~/ov_models/ -d ~/ov_models/
fi

############################################################
#     5.2 run benchmark_app                                #
############################################################
CppBenchmarkFunc()
{
cd $ovPath/openvino/$installDic/samples/cpp/
. build_samples.sh -b .
benchmark_appPath=$ovPath/openvino/$installDic/samples/cpp/intel64/Release/benchmark_app
if command -v $benchmark_appPath; then
  echo "$benchmark_appPath exists"
  $benchmark_appPath -m ~/ov_models/public/$model/FP32/$model.xml -d CPU 
else
  echo "not find $benchmark_appPath"
  exit
 fi
}

if $benchmarkCpp; then
  echo "use benchmark_app c++ version"
  CppBenchmarkFunc
else
  echo "use default benchmark_app python version"
  benchmark_app -m ~/ov_models/public/$model/FP32/$model.xml -d CPU 
fi

echo 'env setup, OV install and evaluation finished.'

conda deactivate && cd 

exit

