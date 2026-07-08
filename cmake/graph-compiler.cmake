include("${CMAKE_CURRENT_LIST_DIR}/llvm.cmake")

get_property(GRAPH_COMPILER_LIBS GLOBAL PROPERTY GRAPH_COMPILER_LIBS)
if (NOT DEFINED GRAPH_COMPILER_LIBS)
    if (DEFINED GraphCompiler_DIR AND EXISTS "${GraphCompiler_DIR}/GraphCompilerTargets.cmake")
      include("${GraphCompiler_DIR}/GraphCompilerTargets.cmake")
    elseif (DEFINED GraphCompiler_ROOT)
      include("${GraphCompiler_ROOT}/lib/cmake/GraphCompiler/GraphCompilerTargets.cmake")
    else()
      find_package(GraphCompiler QUIET)
      if (NOT GraphCompiler_FOUND)
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
        set(GC_DYLINK ${LLVM_DYLINK} CACHE BOOL "" FORCE)
        set(_ov_build_shared_libs ${BUILD_SHARED_LIBS})
        set(BUILD_SHARED_LIBS OFF)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
        FetchContent_MakeAvailable(GraphCompiler)
        set(BUILD_SHARED_LIBS ${_ov_build_shared_libs})
      endif()
    endif()

    if(LLVM_DYLINK)
      set(GRAPH_COMPILER_LIBS GcInterface GraphCompiler)
    else()
      set(GRAPH_COMPILER_LIBS GcInterface MLIRLinalgx GcGpuOclRuntime GcGpuPasses GcGpuOclPasses)
    endif()
    set_property(GLOBAL PROPERTY GRAPH_COMPILER_LIBS ${GRAPH_COMPILER_LIBS})
endif ()

get_target_property(GRAPH_COMPILER_INCLUDES GcInterface INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(GRAPH_COMPILER_COMPILE_OPTIONS GcInterface INTERFACE_COMPILE_OPTIONS)
