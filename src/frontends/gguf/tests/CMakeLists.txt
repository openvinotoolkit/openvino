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
# Explicit source lists (no GLOB: it is being removed from the OpenVINO build). Frontend sources
# are compiled directly into the self-contained test binary; keep this list in sync when adding
# frontend sources or tests.
set(FRONTEND_SRCS
    "${FE_SRC_DIR}/frontend.cpp"
    "${FE_SRC_DIR}/input_model.cpp"
    "${FE_SRC_DIR}/op_table.cpp"
    "${FE_SRC_DIR}/translate_session.cpp"
    "${FE_SRC_DIR}/utils.cpp"
    "${FE_SRC_DIR}/helper_ops/set_rows_op.cpp"
    "${FE_SRC_DIR}/pass/lower_set_rows_stateless.cpp"
    "${FE_SRC_DIR}/quant/gguf_quants.cpp"
    "${FE_SRC_DIR}/quant/weights.cpp"
    "${FE_SRC_DIR}/op/add_id.cpp"
    "${FE_SRC_DIR}/op/argsort.cpp"
    "${FE_SRC_DIR}/op/clamp.cpp"
    "${FE_SRC_DIR}/op/concat.cpp"
    "${FE_SRC_DIR}/op/cont.cpp"
    "${FE_SRC_DIR}/op/cpy.cpp"
    "${FE_SRC_DIR}/op/flash_attn_ext.cpp"
    "${FE_SRC_DIR}/op/gated_delta_net.cpp"
    "${FE_SRC_DIR}/op/get_rows.cpp"
    "${FE_SRC_DIR}/op/glu_geglu.cpp"
    "${FE_SRC_DIR}/op/glu_swiglu.cpp"
    "${FE_SRC_DIR}/op/im2col.cpp"
    "${FE_SRC_DIR}/op/l2_norm.cpp"
    "${FE_SRC_DIR}/op/mul_mat_id.cpp"
    "${FE_SRC_DIR}/op/mulmat.cpp"
    "${FE_SRC_DIR}/op/norm.cpp"
    "${FE_SRC_DIR}/op/pad.cpp"
    "${FE_SRC_DIR}/op/permute.cpp"
    "${FE_SRC_DIR}/op/repeat.cpp"
    "${FE_SRC_DIR}/op/reshape.cpp"
    "${FE_SRC_DIR}/op/rms_norm.cpp"
    "${FE_SRC_DIR}/op/rope.cpp"
    "${FE_SRC_DIR}/op/scale.cpp"
    "${FE_SRC_DIR}/op/set_rows.cpp"
    "${FE_SRC_DIR}/op/softmax.cpp"
    "${FE_SRC_DIR}/op/ssm_conv.cpp"
    "${FE_SRC_DIR}/op/sum_rows.cpp"
    "${FE_SRC_DIR}/op/transpose.cpp"
    "${FE_SRC_DIR}/op/unary_gelu.cpp"
    "${FE_SRC_DIR}/op/unary_silu.cpp"
    "${FE_SRC_DIR}/op/view.cpp"
    "${FE_SRC_DIR}/op/weight.cpp"
)

set(TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/test_dequant_vs_ggml.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_extensions.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_ops.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_weights.cpp"
)

# test_data/ lives next to this CMakeLists.txt; expose the absolute path to the
# C++ test binary so load_npy() can find files without hard-coding a path.
set(GGUF_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_data")

ov_add_test_target(
    NAME ${TARGET_NAME}
    SOURCES ${TEST_SRCS} ${FRONTEND_SRCS}
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

# The sources are compiled into the test binary, so the GGUF_FRONTEND_API
# visibility macros must resolve to local definitions (as when building the
# frontend library itself) rather than dllimport.
target_compile_definitions(${TARGET_NAME} PRIVATE openvino_gguf_frontend_EXPORTS)

# The frontend uses ov::parallel_for (openvino/core/parallel.hpp), which needs the
# threading (TBB) include directories.
ov_set_threading_interface_for(${TARGET_NAME})

# The frontend sources contain file-internal helpers without separate declarations; relax only
# that warning so the test builds with -Werror. (Narrowing conversions are fixed in the sources
# themselves, so -Wno-conversion / C4244 / C4267 are intentionally not suppressed here.)
if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    set_source_files_properties(${FRONTEND_SRCS} PROPERTIES
        COMPILE_OPTIONS "-Wno-missing-declarations")
endif()

# Inject the test data directory path as a compile-time string constant.
target_compile_definitions(${TARGET_NAME} PRIVATE
    TEST_DATA_DIR="${GGUF_TEST_DATA_DIR}"
)

ov_build_target_faster(${TARGET_NAME} PCH)
