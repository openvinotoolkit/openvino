include_guard()
include("${CMAKE_CURRENT_LIST_DIR}/llvm.cmake")

find_package(GraphCompiler QUIET CONFIG)

if (NOT GraphCompiler_FOUND)
  option(GRAPH_COMPILER_DYLINK "Use dynamic linking with GraphCompiler" OFF)
  set(GRAPH_COMPILER_REPO "https://github.com/intel-sandbox/graph-compiler" CACHE STRING "GraphCompiler repository URL")
  set(GRAPH_COMPILER_TAG "main" CACHE STRING "GraphCompiler git tag/branch")
  message(STATUS "GraphCompiler not found, fetching from: ${GRAPH_COMPILER_REPO}")
  include(FetchContent)
  FetchContent_Declare(
      GraphCompiler
      GIT_REPOSITORY ${GRAPH_COMPILER_REPO}
      GIT_TAG ${GRAPH_COMPILER_TAG}
      GIT_SHALLOW TRUE
  )
  set(GC_ENABLE_TEST OFF CACHE BOOL "" FORCE)
  set(GC_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
  set(GC_ENABLE_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  set(GC_DYLINK ${GRAPH_COMPILER_DYLINK} CACHE BOOL "" FORCE)
  set(_ov_build_shared_libs ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS ${GRAPH_COMPILER_DYLINK})
  FetchContent_MakeAvailable(GraphCompiler)
  set(BUILD_SHARED_LIBS ${_ov_build_shared_libs})
endif()
