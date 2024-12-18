get_property(GRAPH_COMPILER_LIBS GLOBAL PROPERTY GRAPH_COMPILER_LIBS)
if (NOT DEFINED GRAPH_COMPILER_LIBS)
    # The FetchContent_Declare(FIND_PACKAGE_ARGS) is supported since CMake 3.24. For the prior
    # versions, using find_package() first. If the package is not found, then using FetchContent.
    if (CMAKE_VERSION VERSION_LESS "3.24")
        find_package(GraphCompiler QUIET)
    else ()
        set(GC_FETCH_CONTENT_ARGS FIND_PACKAGE_ARGS NAMES GraphCompiler)
    endif ()

    if (NOT GraphCompiler_FOUND)
        include(FetchContent)

        FetchContent_Declare(
                GC
                GIT_REPOSITORY https://github.com/intel/graph-compiler.git
                GIT_TAG main
                ${GC_FETCH_CONTENT_ARGS}
        )

        set(GC_ENABLE_IMEX ${ENABLE_INTEL_GPU})
        set(GC_ENABLE_TOOLS OFF)
        set(GC_ENABLE_TEST OFF)
        set(GC_ENABLE_DNNL_API OFF)
        set(GC_ENABLE_LEGACY OFF)
        set(GC_ENABLE_BINDINGS_PYTHON OFF)
        set(OV_BUILD_SHARED_LIBS_TMP ${BUILD_SHARED_LIBS})
        set(BUILD_SHARED_LIBS OFF)

        FetchContent_GetProperties(GC)
        if(NOT GC_POPULATED)
            FetchContent_Populate(GC)
        endif()
        if(NOT DEFINED LLVM_DIR OR NOT DEFINED MLIR_DIR)
            execute_process(
                COMMAND /bin/bash ./scripts/compile.sh --dev --llvm --imex
                WORKING_DIRECTORY ${gc_SOURCE_DIR}
            )

            SET(LLVM_INST_PATH "${gc_SOURCE_DIR}/externals/llvm-project/build")
            SET(LLVM_DIR "${LLVM_INST_PATH}/lib/cmake/llvm")
            SET(MLIR_DIR "${LLVM_INST_PATH}/lib/cmake/mlir")
            add_definitions(-DMLIR_RUNNER_UTILS_PATH=${LLVM_INST_PATH}/lib)
            add_subdirectory(${gc_SOURCE_DIR} ${gc_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()

        FetchContent_MakeAvailable(GC)
        set(BUILD_SHARED_LIBS ${OV_BUILD_SHARED_LIBS_TMP})
        FetchContent_GetProperties(GC BINARY_DIR gc_BINARY_DIR)
        ov_cpack_add_component(GcCpuRuntime HIDDEN)
        set(GC_CPU_RUNTIME_PATH ${gc_BINARY_DIR}/lib/libGcCpuRuntime.so)
        install(FILES ${GC_CPU_RUNTIME_PATH} DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT GcCpuRuntime)
        # a hack to not bother with actual file extension
        install(FILES ${GC_CPU_RUNTIME_PATH}.20.0git DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT GcCpuRuntime)
    endif ()

    set(GRAPH_COMPILER_LIBS
            GcInterface
            GcJitWrapper
            GcCpuRuntime
    )

    if (ENABLE_INTEL_GPU)
        list(APPEND GRAPH_COMPILER_LIBS GcGpuOclRuntime)
    endif()

    set_property(GLOBAL PROPERTY GRAPH_COMPILER_LIBS ${GRAPH_COMPILER_LIBS})
endif ()

get_target_property(GRAPH_COMPILER_INCLUDES GcInterface INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(GRAPH_COMPILER_COMPILE_OPTIONS GcInterface INTERFACE_COMPILE_OPTIONS)
