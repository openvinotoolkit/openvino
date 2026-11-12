# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# ggml -- https://github.com/ggml-org/ggml (MIT)
#
# Downloaded on demand rather than vendored as a submodule: it is only needed when
# ENABLE_GGML_CGRAPH_LOADER is set, and the Vulkan backend alone is ~43 MB of precompiled SPIR-V, so
# there is no reason to make every clone pay for it.
#
# This is the STANDALONE ggml project, not llama.cpp. ggml is a self-contained CMake project with
# its own semver and tagged releases that exports ggml-config.cmake, so llama.cpp does not enter
# the dependency graph -- no libllama, no build-order coupling to that repo.
#
# An existing install takes precedence: pass -Dggml_DIR=<prefix>/lib/cmake/ggml to use a
# system/prebuilt ggml and skip the download entirely.

find_package(ggml QUIET)

if(ggml_FOUND)
    message(STATUS "ggml: using pre-installed ggml at ${ggml_DIR}")
    # find_package ran in THIS directory scope, so the imported targets it created are invisible
    # to src/ggml_cgraph_loader. Promote them; ggml-config.cmake declares real IMPORTED targets (not
    # ALIASes), so this is legal here and only here.
    foreach(_tgt ggml::ggml ggml::ggml-base)
        if(TARGET ${_tgt})
            set_target_properties(${_tgt} PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
    endforeach()
    return()
endif()

# Pinned. Bumping this is a deliberate act: ggml syncs its backends FROM llama.cpp, so a bump can
# change kernel behaviour. Re-run the archprobe op sweep after changing it.
set(OV_GGML_GIT_TAG "v0.23.0")

if(ENABLE_GGML_VULKAN)
    # ggml compiles its Vulkan shaders to SPIR-V at build time and needs a GLSL compiler to do
    # it. Fail here with something actionable rather than deep inside the fetched project.
    find_program(OV_GLSLC_EXECUTABLE NAMES glslc)
    if(NOT OV_GLSLC_EXECUTABLE)
        message(FATAL_ERROR
            "ENABLE_GGML_VULKAN=ON requires glslc to compile the Vulkan shaders, but it was not "
            "found. Install it (Ubuntu/Debian: 'apt install glslc' or the Vulkan SDK; Fedora: "
            "'dnf install glslc'), or configure with -DENABLE_GGML_VULKAN=OFF to build the ggml "
            "CPU backend only.")
    endif()
    message(STATUS "ggml: Vulkan backend enabled (glslc: ${OV_GLSLC_EXECUTABLE})")
endif()

# ggml's options are plain CMake options, so force them in the cache before the project is added.
set(GGML_VULKAN        ${ENABLE_GGML_VULKAN} CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF                 CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF                 CACHE BOOL "" FORCE)
set(GGML_BUILD_TOOLS    OFF                 CACHE BOOL "" FORCE)
# ggml defaults to -march=native, which would make the build non-portable and can break
# cross-compiles and CI runners that differ from the build host.
set(GGML_NATIVE         OFF                 CACHE BOOL "" FORCE)
# Unloading the plugin also unloads MSVC's OpenMP runtime (vcomp140.dll) while its worker threads
# are still parked in it, and they crash; ggml's own thread pool is joined before compute returns.
if(WIN32)
    set(GGML_OPENMP     OFF                 CACHE BOOL "" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        ${OV_GGML_GIT_TAG}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    # 1) PTQ1_0, PrismML's ternary quant type (Ternary-Bonsai GGUFs): CPU kernels only, small and additive.
    # 2) GGML_MAX_NAME 64 -> 128: some GGUFs (Qwen-Image-2.1 from unsloth) carry tensor names up to 71 characters.
    PATCH_COMMAND  ${CMAKE_COMMAND} -DOV_GGML_GIT_TAG=${OV_GGML_GIT_TAG}
                       "-DPATCH=${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-ptq1_0.patch|${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-max-name.patch"
                       -P ${CMAKE_CURRENT_SOURCE_DIR}/patches/apply_patch.cmake)

message(STATUS "ggml: fetching ${OV_GGML_GIT_TAG} (set -Dggml_DIR=... to use a prebuilt one)")
FetchContent_MakeAvailable(ggml)

# FetchContent gives bare targets (ggml, ggml-base); the loader links the namespaced names that
# find_package would have provided, so alias them and keep both paths identical.
foreach(_tgt ggml ggml-base)
    if(TARGET ${_tgt} AND NOT TARGET ggml::${_tgt})
        add_library(ggml::${_tgt} ALIAS ${_tgt})
    endif()
endforeach()

# Co-locate the ggml libraries with OpenVINO's own, so the loader finds its backends at runtime
# without LD_LIBRARY_PATH: ggml_backend_load_all() dlopens ggml-cpu / ggml-vulkan by filename and
# looks beside the loading library. Also install them, since the loader is useless without them.
foreach(_tgt ggml ggml-base ggml-cpu ggml-vulkan)
    if(TARGET ${_tgt})
        set_target_properties(${_tgt} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${OV_LIBRARY_OUTPUT_DIRECTORY}"
            RUNTIME_OUTPUT_DIRECTORY "${OV_LIBRARY_OUTPUT_DIRECTORY}")
        install(TARGETS ${_tgt}
                LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${OV_CPACK_COMP_CORE}
                RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${OV_CPACK_COMP_CORE})
    endif()
endforeach()
