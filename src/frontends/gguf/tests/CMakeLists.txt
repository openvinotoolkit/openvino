# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_gguf_frontend_tests)

# The test binary is self-contained: it compiles the frontend sources directly
# instead of linking the installed shared library.  This keeps the frontend out
# of the OpenVINO frontend search directory (get_ov_lib_path()), which
# FrontEndManager scans and tries to load every "openvino_*_frontend" library
# from; the MVP frontend has no plugin entry points, so being discoverable there
# would break the generic FrontEndManager tests.
set(FE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
file(GLOB_RECURSE FRONTEND_SRCS "${FE_SRC_DIR}/*.cpp")

# test_data/ lives next to this CMakeLists.txt; expose the absolute path to the
# C++ test binary so load_npy() can find files without hard-coding a path.
set(GGUF_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_data")

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        gtest
        gtest_main
        common_test_utils
        openvino::runtime
        openvino::runtime::dev
        openvino::offline_transformations
        openvino::cnpy
    INCLUDES
        "${CMAKE_CURRENT_SOURCE_DIR}/../src"
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    ADD_CLANG_FORMAT
    LABELS
        OV UNIT GGUF_FE
)

# Install the .npy reference data next to the test binary so the installed test
# (run in CI from the tests component) can find it via getExecutableDirectory().
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test_data"
        DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
        FILES_MATCHING PATTERN "*.npy")

# Add the frontend sources explicitly (ov_add_test_target only scans ROOT for .cpp).
target_sources(${TARGET_NAME} PRIVATE ${FRONTEND_SRCS})

# The sources are compiled into the test binary, so the GGUF_FRONTEND_API
# visibility macros must resolve to local definitions (as when building the
# frontend library itself) rather than dllimport.
target_compile_definitions(${TARGET_NAME} PRIVATE openvino_gguf_frontend_EXPORTS)

# The frontend uses ov::parallel_for (openvino/core/parallel.hpp), which needs the
# threading (TBB) include directories.
ov_set_threading_interface_for(${TARGET_NAME})

# The frontend sources are ported verbatim from llama.cpp and contain file-internal
# helpers without separate declarations and narrowing conversions; relax the matching
# warnings (the frontend library does the same) so the test builds with -Werror.
if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    set_source_files_properties(${FRONTEND_SRCS} PROPERTIES
        COMPILE_OPTIONS "-Wno-missing-declarations;-Wno-conversion")
elseif(MSVC)
    set_source_files_properties(${FRONTEND_SRCS} PROPERTIES COMPILE_OPTIONS "/wd4244;/wd4267")
endif()

# Inject the test data directory path as a compile-time string constant.
target_compile_definitions(${TARGET_NAME} PRIVATE
    TEST_DATA_DIR="${GGUF_TEST_DATA_DIR}"
)

ov_build_target_faster(${TARGET_NAME} PCH)
