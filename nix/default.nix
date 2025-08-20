{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
, python3
, pkg-config
, git
, libarchive
, patchelf
, autoPatchelfHook
, addDriverRunpath
, scons
, shellcheck
, opencv
, protobuf
, flatbuffers
, pugixml
, snappy
, tbb
, gflags
, level-zero
, libusb1
, libxml2
, ocl-icd
, tree
, cudaSupport ? false
, cudaPackages ? {}
, tbbbind
, tbbbind_version
}:

let
  # Conditional test building - can be overridden by users
  enableTests = false;
  
  # Helper function for cmake boolean flags
  cmakeBool = name: value: "-D${name}=${if value then "ON" else "OFF"}";
in

stdenv.mkDerivation rec {
  pname = "openvino";
  version = "2025.4.0";

  src = fetchFromGitHub {
    owner = "openvinotoolkit";
    repo = "openvino";
    rev = "master"; # Use master for latest development version
    fetchSubmodules = true;
    hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; # Will be updated by nix-prefetch-git
  };

  outputs = [
    "out"
    "python"
  ];

  nativeBuildInputs = [
    addDriverRunpath
    autoPatchelfHook
    cmake
    git
    libarchive
    patchelf
    pkg-config
    python3
    scons
    shellcheck
  ] ++ lib.optionals cudaSupport [
    cudaPackages.cuda_nvcc
  ];

  postPatch = ''
    mkdir -p temp/tbbbind_${tbbbind_version}
    pushd temp/tbbbind_${tbbbind_version}
    bsdtar -xf ${tbbbind}
    echo "${tbbbind.url}" > ie_dependency.info
    popd
  '';

  dontUseSconsCheck = true;
  dontUseSconsBuild = true;
  dontUseSconsInstall = true;

  cmakeFlags = [
    "-Wno-dev"
    "-DCMAKE_MODULE_PATH:PATH=${placeholder "out"}/lib/cmake"
    "-DCMAKE_PREFIX_PATH:PATH=${placeholder "out"}"

    "-DCMAKE_CXX_FLAGS=-Wno-odr"
    "-DCMAKE_C_FLAGS=-Wno-odr"

    "-DOpenCV_DIR=${lib.getLib opencv}/lib/cmake/opencv4/"
    "-DProtobuf_LIBRARIES=${protobuf}/lib/libprotobuf${stdenv.hostPlatform.extensions.sharedLibrary}"
    "-DPython_EXECUTABLE=${python3.interpreter}"

    (cmakeBool "CMAKE_VERBOSE_MAKEFILE" true)
    (cmakeBool "NCC_STYLE" false) 
    (cmakeBool "ENABLE_CPPLINT" false)
    
    # FIXED: Proper test configuration to prevent build conflicts
    (cmakeBool "ENABLE_TESTS" enableTests)
    (cmakeBool "ENABLE_FUNCTIONAL_TESTS" enableTests)
    (cmakeBool "BUILD_TESTING" enableTests)
    (cmakeBool "ENABLE_SAMPLES" false)

    # features
    (cmakeBool "ENABLE_INTEL_CPU" stdenv.hostPlatform.isx86_64)
    (cmakeBool "ENABLE_INTEL_GPU" true)
    (cmakeBool "ENABLE_INTEL_NPU" stdenv.hostPlatform.isx86_64)
    (cmakeBool "ENABLE_JS" false)
    (cmakeBool "ENABLE_LTO" true)
    (cmakeBool "ENABLE_ONEDNN_FOR_GPU" false)
    (cmakeBool "ENABLE_OPENCV" true)
    (cmakeBool "ENABLE_OV_JAX_FRONTEND" false)
    (cmakeBool "ENABLE_PYTHON" true)

    # system libs
    (cmakeBool "ENABLE_SYSTEM_FLATBUFFERS" true)
    (cmakeBool "ENABLE_SYSTEM_OPENCL" true)
    (cmakeBool "ENABLE_SYSTEM_PROTOBUF" false)
    (cmakeBool "ENABLE_SYSTEM_PUGIXML" true)
    (cmakeBool "ENABLE_SYSTEM_SNAPPY" true)
    (cmakeBool "ENABLE_SYSTEM_TBB" true)
  ];

  autoPatchelfIgnoreMissingDeps = [
    "libngraph_backend.so"
  ];

  buildInputs = [
    flatbuffers
    gflags
    level-zero
    libusb1
    libxml2
    ocl-icd
    opencv
    pugixml
    snappy
    tbb
    tree
  ] ++ lib.optionals cudaSupport [
    cudaPackages.cuda_cudart
  ];

  enableParallelBuilding = true;

  preInstallPhase = ''
    echo "preInstall"
    tree .
  '';

  postInstall = ''
    mkdir -p $python
    mv $out/python/* $python/
    rmdir $out/python
  '';

  postFixup = ''
    # Link to OpenCL
    find $out -type f \( -name '*.so' -or -name '*.so.*' \) | while read lib; do
      addDriverRunpath "$lib"
    done
  '';

  meta = with lib; {
    changelog = "https://github.com/openvinotoolkit/openvino/releases";
    description = "OpenVINO™ Toolkit - Open source toolkit for optimizing and deploying AI inference";
    longDescription = ''
      This toolkit allows developers to deploy pre-trained deep learning models through a high-level C++ Inference Engine API integrated with application logic.

      This open source version includes several components: namely Model Optimizer, nGraph and Inference Engine, as well as CPU, GPU, MYRIAD,
      multi device and heterogeneous plugins to accelerate deep learning inferencing on Intel® CPUs and Intel® Processor Graphics.
      It supports pre-trained models from the Open Model Zoo, along with 100+ open source and public models in popular formats such as Caffe*, TensorFlow*, MXNet* and ONNX*.
    '';
    homepage = "https://docs.openvino.ai/";
    license = with licenses; [ asl20 ];
    platforms = platforms.all;
    broken = stdenv.hostPlatform.isDarwin; # Cannot find macos sdk
    maintainers = with maintainers; [ ];
  };
}
