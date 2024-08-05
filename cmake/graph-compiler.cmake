get_property(GRAPH_COMPILER_LIBS GLOBAL PROPERTY GRAPH_COMPILER_LIBS)
if (NOT DEFINED GRAPH_COMPILER_LIBS)
    # The FetchContent_Declare(FIND_PACKAGE_ARGS) is supported since CMake 3.24. For the prior
    # versions, using find_package() first. If the package is not found, then using FetchContent.
    if (CMAKE_VERSION VERSION_LESS "3.24")
        find_package(GraphCompiler QUIET)
    endif ()

    if (NOT GraphCompiler_FOUND)
        include(FetchContent)

        FetchContent_Declare(
                GC
                GIT_REPOSITORY https://github.com/intel/graph-compiler.git
                GIT_TAG main
                FIND_PACKAGE_ARGS NAMES GraphCompiler
        )

        set(GC_ENABLE_OPT OFF)
        set(GC_ENABLE_TEST OFF)
        set(GC_ENABLE_DNNL OFF)
        set(GC_ENABLE_LEGACY OFF)
        set(GC_ENABLE_BINDINGS_PYTHON OFF)
        set(OV_BUILD_SHARED_LIBS_TMP ${BUILD_SHARED_LIBS})
        set(BUILD_SHARED_LIBS OFF)
        FetchContent_MakeAvailable(GC)
        set(BUILD_SHARED_LIBS ${OV_BUILD_SHARED_LIBS_TMP})
    endif ()

    set(GRAPH_COMPILER_LIBS
            GcInterface
            GcJitWrapper
            GcCpuRuntime
    )
    set_property(GLOBAL PROPERTY GRAPH_COMPILER_LIBS ${GRAPH_COMPILER_LIBS})
endif ()

get_target_property(GRAPH_COMPILER_INCLUDES GcInterface INTERFACE_INCLUDE_DIRECTORIES)
