# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "D:/OPENVINO/openvino/src/plugins/intel_gpu/thirdparty/onednn_gpu"
  "D:/OPENVINO/openvino/src/plugins/intel_gpu/thirdparty/onednn_gpu_build"
  "D:/OPENVINO/openvino/src/plugins/intel_gpu/thirdparty/onednn_gpu_install"
  "D:/OPENVINO/openvino/src/plugins/intel_gpu/thirdparty/onednn_gpu_root/tmp"
  "D:/OPENVINO/openvino/src/plugins/intel_gpu/thirdparty/onednn_gpu_root/src/onednn_gpu_build-stamp"
  "D:/OPENVINO/openvino/src/plugins/intel_gpu/thirdparty/onednn_gpu_root/src"
  "D:/OPENVINO/openvino/src/plugins/intel_gpu/thirdparty/onednn_gpu_root/src/onednn_gpu_build-stamp"
)

set(configSubDirs Debug;Release;MinSizeRel;RelWithDebInfo)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/OPENVINO/openvino/src/plugins/intel_gpu/thirdparty/onednn_gpu_root/src/onednn_gpu_build-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/OPENVINO/openvino/src/plugins/intel_gpu/thirdparty/onednn_gpu_root/src/onednn_gpu_build-stamp${cfgdir}") # cfgdir has leading slash
endif()
