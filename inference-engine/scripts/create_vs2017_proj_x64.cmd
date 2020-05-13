@echo off
:: Copyright (C) 2018-2020 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

pushd ..\..
if not exist "vs2017x64" (
	mkdir "vs2017x64"
)

cmake -E chdir "vs2017x64" cmake -G "Visual Studio 15 2017 Win64" -T "Intel C++ Compiler 18.0" -DOS_FOLDER=ON ^
	-DENABLE_MYRIAD=OFF -DENABLE_VPU=OFF -DENABLE_GNA=ON -DENABLE_CLDNN=OFF ^
	-DENABLE_OPENCV=ON -DENABLE_MKL_DNN=ON ^
	-DVERBOSE_BUILD=ON -DENABLE_TESTS=ON -DTHREADING=TBB ..


chdir
cd "vs2017x64\thirdparty\"
"C:\Program Files (x86)\Common Files\Intel\shared files\ia32\Bin\ICProjConvert180.exe" mkldnn.vcxproj /IC 

chdir
cd "..\src\mkldnn_plugin"
"C:\Program Files (x86)\Common Files\Intel\shared files\ia32\Bin\ICProjConvert180.exe" MKLDNNPlugin.vcxproj /IC 
"C:\Program Files (x86)\Common Files\Intel\shared files\ia32\Bin\ICProjConvert180.exe" test_MKLDNNPlugin.vcxproj /IC 

chdir
cd "..\..\tests\unit"
"C:\Program Files (x86)\Common Files\Intel\shared files\ia32\Bin\ICProjConvert180.exe" InferenceEngineUnitTests.vcxproj /IC 


popd
pause
