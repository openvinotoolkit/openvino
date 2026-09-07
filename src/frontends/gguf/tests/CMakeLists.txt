# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_gguf_frontend_tests)

if(NOT BUILD_SHARED_LIBS)
    # This target compiles the frontend's own sources directly (white-box access to internal
    # helpers) and links openvino::runtime, which in static builds already bundles every
    # registered frontend via ov_target_link_frontends() -- the duplicate frontend.cpp objects
    # then conflict at link time. No other frontend's tests compile their own sources this way,
    # so skip only here rather than changing that shared static-build mechanism.
    return()
endif()

# Compile the frontend sources directly into the test binary instead of linking the
# installed library. Explicit source lists (no GLOB); keep in sync when adding sources.
set(FE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
set(FRONTEND_SRCS
    "${FE_SRC_DIR}/frontend.cpp"
    "${FE_SRC_DIR}/input_model.cpp"
    "${FE_SRC_DIR}/op_table.cpp"
    "${FE_SRC_DIR}/translate_session.cpp"
    "${FE_SRC_DIR}/utils.cpp"
    "${FE_SRC_DIR}/helper_ops/set_rows_op.cpp"
    "${FE_SRC_DIR}/pass/adapt_to_genai.cpp"
    "${FE_SRC_DIR}/pass/lower_set_rows_stateless.cpp"
    "${FE_SRC_DIR}/pass/make_stateful.cpp"
    "${FE_SRC_DIR}/pass/prune_orphaned_parameters.cpp"
    "${FE_SRC_DIR}/tokenizer_metadata.cpp"
    "${FE_SRC_DIR}/quant/gguf.cpp"
    "${FE_SRC_DIR}/quant/gguf_quants.cpp"
    "${FE_SRC_DIR}/quant/weights.cpp"
    "${FE_SRC_DIR}/builder/gguf_builder.cpp"
    "${FE_SRC_DIR}/builder/gguf_builder_decoder.cpp"
    "${FE_SRC_DIR}/builder/graph_emitter.cpp"
    "${FE_SRC_DIR}/builder/arch_registry.cpp"
    "${FE_SRC_DIR}/builder/decoder_config.cpp"
    "${FE_SRC_DIR}/builder/model_kind.cpp"
    "${FE_SRC_DIR}/builder/arch/decoder_builder.cpp"
    "${FE_SRC_DIR}/builder/blocks/attention.cpp"
    "${FE_SRC_DIR}/builder/blocks/common.cpp"
    "${FE_SRC_DIR}/builder/blocks/ffn.cpp"
    "${FE_SRC_DIR}/builder/blocks/gated_delta_net.cpp"
    "${FE_SRC_DIR}/builder/blocks/qkv_repack.cpp"
    "${FE_SRC_DIR}/op/add_id.cpp"
    "${FE_SRC_DIR}/op/argsort.cpp"
    "${FE_SRC_DIR}/op/clamp.cpp"
    "${FE_SRC_DIR}/op/concat.cpp"
    "${FE_SRC_DIR}/op/cont.cpp"
    "${FE_SRC_DIR}/op/cpy.cpp"
    "${FE_SRC_DIR}/op/cumsum.cpp"
    "${FE_SRC_DIR}/op/diag.cpp"
    "${FE_SRC_DIR}/op/div.cpp"
    "${FE_SRC_DIR}/op/fill.cpp"
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
    "${FE_SRC_DIR}/op/set.cpp"
    "${FE_SRC_DIR}/op/set_rows.cpp"
    "${FE_SRC_DIR}/op/softmax.cpp"
    "${FE_SRC_DIR}/op/sqr.cpp"
    "${FE_SRC_DIR}/op/ssm_conv.cpp"
    "${FE_SRC_DIR}/op/top_k.cpp"
    "${FE_SRC_DIR}/op/tri.cpp"
    "${FE_SRC_DIR}/op/sum_rows.cpp"
    "${FE_SRC_DIR}/op/transpose.cpp"
    "${FE_SRC_DIR}/op/unary_elu.cpp"
    "${FE_SRC_DIR}/op/unary_gelu.cpp"
    "${FE_SRC_DIR}/op/unary_silu.cpp"
    "${FE_SRC_DIR}/op/view.cpp"
    "${FE_SRC_DIR}/op/weight.cpp"
)

set(TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/test_adapt_to_genai.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_arch_conversion.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_dequant_vs_ggml.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_extensions.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_op_coverage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_ops.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_weights.cpp"
)

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
        # frontend.cpp (compiled into the self-contained test binary) includes the native
        # .gguf load path (get_path_from_any in openvino/frontend/common/path_util.hpp).
        "${OpenVINO_SOURCE_DIR}/src/frontends/common/include"
        "${OpenVINO_SOURCE_DIR}/src/frontends/common/dev_api"
    ADD_CLANG_FORMAT
    LABELS
        OV UNIT GGUF_FE
)

# Install the reference data next to the test binary (found via getExecutableDirectory()):
# *.npy op/dequant references, plus the per-architecture fixture manifest.
#
# The manifest's companion *.gguf.hdr fixtures are deliberately NOT here: they are generated from
# llama.cpp by tests/gen_arch_fixtures.py into this same directory (see the "Generate GGUF arch
# fixtures" step in .github/workflows/job_build_linux.yml).  The manifest itself IS installed --
# it carries the reviewed per-architecture expectation for each fixture, which is source, not
# generated output.  Where no fixtures were generated the arch suite skips itself; see
# test_arch_conversion.cpp.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test_data"
        DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
        FILES_MATCHING
            PATTERN "*.npy"
            PATTERN "manifest.txt")

# Sources compiled into the binary: resolve the visibility macros to local definitions, not dllimport.
target_compile_definitions(${TARGET_NAME} PRIVATE openvino_gguf_frontend_EXPORTS)

# ov::parallel_for needs the threading (TBB) include directories.
ov_set_threading_interface_for(${TARGET_NAME})

# Frontend sources have file-internal helpers without separate declarations; relax only that warning.
if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    set_source_files_properties(${FRONTEND_SRCS} PROPERTIES
        COMPILE_OPTIONS "-Wno-missing-declarations")
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE
    TEST_DATA_DIR="${GGUF_TEST_DATA_DIR}"
)

ov_build_target_faster(${TARGET_NAME} PCH)
