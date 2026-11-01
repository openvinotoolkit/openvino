# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Experimental GPU plugin path: BlockSparseAttention / DraftBlockMask primitives, graph nodes, CM
# kernel implementations, plugin ops, and hooks. Only included when ENABLE_OV_EXPERIMENTAL=ON (see
# the parent CMakeLists.txt). Everything here is confined to the experimental shadow tree so that
# upstream sources (incl. the upstream SourcesDB that embeds the upstream CM kernel sources) stay
# byte-identical when ENABLE_OV_EXPERIMENTAL=OFF (see scripts/check_experimental_parity.sh).
#
# The graph library is built once per runtime stack (openvino_intel_gpu_graph for the ZE/single
# twin, and additionally openvino_intel_gpu_graph_ocl for the OCL twin in a COMBINED build), so the
# experimental graph/plugin sources and include dirs are applied to each builder suffix below. The
# CM kernel codegen runs a single time (one .inc output), independent of the number of graph twins.

if(NOT ENABLE_OV_EXPERIMENTAL)
    message(FATAL_ERROR "src/plugins/intel_gpu/src/experimental must only be configured when ENABLE_OV_EXPERIMENTAL is ON")
endif()

# Experimental sub-tree root (src/plugins/intel_gpu/src/experimental).
set(OV_GPU_EXP_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Core experimental op headers (ov::op::experimental::BlockSparseAttention / DraftBlockMask).
set(OV_CORE_EXP_INCLUDE "${PROJECT_SOURCE_DIR}/src/core/experimental/include")
set(OV_CORE_EXP_SHAPE_INFER_INCLUDE "${PROJECT_SOURCE_DIR}/src/core/experimental/shape_inference/include")

set(EXP_INCLUDE_DIRS
    ${OV_GPU_EXP_DIR}
    ${MAIN_DIR}/src
    ${OV_CORE_EXP_INCLUDE}
    ${OV_CORE_EXP_SHAPE_INFER_INCLUDE})

# -----------------------------------------------------------------------------
# Graph-side sources -> openvino_intel_gpu_graph<SUFFIX>
# -----------------------------------------------------------------------------
# The graph primitives/inst/layout logic and the primitive-type registry live in the graph
# library. The `ov::intel_gpu::Registry<cldnn::...>` specializations (defined in registry/*.cpp and
# declared in experimental/registry/registry.hpp) are linked into the graph target so that impl
# selection picks up the experimental CM managers.
set(OV_GPU_EXP_GRAPH_SOURCES
    ${OV_GPU_EXP_DIR}/graph/block_sparse_attention.cpp
    ${OV_GPU_EXP_DIR}/graph/draft_block_mask.cpp
    ${OV_GPU_EXP_DIR}/registry/block_sparse_attention_impls.cpp
    ${OV_GPU_EXP_DIR}/registry/draft_block_mask_impls.cpp)

# -----------------------------------------------------------------------------
# Plugin-side sources -> openvino_intel_gpu_plugin<SUFFIX>
# -----------------------------------------------------------------------------
# The plugin op factories (ov::op::experimental::*) and the guarded entry hooks. `register_primitives()`
# is invoked from Plugin::Plugin() under `#if defined(OPENVINO_EXPERIMENTAL)`.
set(OV_GPU_EXP_PLUGIN_SOURCES
    ${OV_GPU_EXP_DIR}/hooks.cpp
    ${OV_GPU_EXP_DIR}/plugin/op_block_sparse_attention.cpp
    ${OV_GPU_EXP_DIR}/plugin/op_draft_block_mask.cpp)

# -----------------------------------------------------------------------------
# CM kernel codegen + CM implementation sources (only when CM support is enabled)
# -----------------------------------------------------------------------------
if(ENABLE_CM_FOR_GPU)
    # Repository path to the stock CM kernel code generator. We reuse the upstream generator script
    # (read-only) but emit a *separate* `experimental_cm_kernel_sources.inc` so the upstream
    # gpu_cm_kernel_sources.inc / SourcesDB remain untouched (the parity gate verified these are
    # listed explicitly, not globbed, so no new .cm/.hpp can drift in silently).
    set(OV_GPU_CM_CODEGEN_SCRIPT "${PROJECT_SOURCE_DIR}/src/plugins/intel_gpu/src/graph/common_utils/kernels_db_gen.py")
    set(OV_GPU_EXP_CODEGEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/cm_codegen")
    set(OV_GPU_EXP_CM_KERNEL_DIR "${OV_GPU_EXP_CODEGEN_DIR}/cm_kernels")
    set(OV_GPU_EXP_CM_HEADER_DIR "${OV_GPU_EXP_CM_KERNEL_DIR}/include")
    set(OV_GPU_EXP_CM_INC "${OV_GPU_EXP_CODEGEN_DIR}/experimental_cm_kernel_sources.inc")

    # Experimental CM kernels (graph/impls/cm/**/*.cm), relative to OV_GPU_EXP_DIR, listed
    # explicitly (never GLOB) so the parity gate sees a deterministic, reviewable source set.
    set(OV_GPU_EXP_CM_KERNELS
        graph/impls/cm/block_sparse_attention/block_sparse_attention_prefill.cm
        graph/impls/cm/draft_block_mask/draft_block_mask_pool.cm
        graph/impls/cm/draft_block_mask/draft_block_mask_score_softmax.cm
        graph/impls/cm/draft_block_mask/draft_block_mask_topk_mask.cm)
    # Headers the kernels `#include` and the codegen inlines into the .inc.
    set(OV_GPU_EXP_CM_HEADERS
        graph/impls/cm/include/cm_sdpa_common.hpp
        graph/impls/cm/include/block_sparse_attention/attention_layout.hpp)

    # Upstream (non-experimental) headers the experimental CM kernels reference via local
    # `#include "..."`. They are staged next to the experimental headers so the include
    # verifier and the codegen can resolve them (graph/impls/cm/include lives upstream, NOT in
    # the experimental shadow tree, so it is never globbed in -- it is listed explicitly here).
    set(OV_GPU_EXP_CM_UPSTREAM_HEADERS
        "${PROJECT_SOURCE_DIR}/src/plugins/intel_gpu/src/graph/impls/cm/include/cm_attention_common.hpp"
        "${PROJECT_SOURCE_DIR}/src/plugins/intel_gpu/src/graph/impls/cm/include/cm_qq_bias_mask.hpp")

    # Resolve the relative kernel/header lists to absolute paths under OV_GPU_EXP_DIR. This also
    # fixes the historical path-prefix bug where an absolute header path would otherwise be
    # double-prefixed.
    set(FULL_EXP_CM_KERNELS "")
    foreach(k ${OV_GPU_EXP_CM_KERNELS})
        list(APPEND FULL_EXP_CM_KERNELS "${OV_GPU_EXP_DIR}/${k}")
    endforeach()

    set(FULL_EXP_CM_HEADERS "")
    foreach(h ${OV_GPU_EXP_CM_HEADERS})
        list(APPEND FULL_EXP_CM_HEADERS "${OV_GPU_EXP_DIR}/${h}")
    endforeach()

    # Single atomic codegen command: reset the staging dir, stage kernels + experimental/upstream
    # headers, statically verify every local include resolves (fail loudly otherwise), then run
    # kernels_db_gen.py. Because it is one command with one OUTPUT, a failure in any stage cannot
    # leave a stale .inc behind.
    #
    # Note: the experimental headers are staged via copy_directory (preserving the relative
    # subdirectory under graph/impls/cm/include -- e.g. block_sparse_attention/attention_layout.hpp
    # -- which kernel `#include "..."` paths reference), while the upstream headers are added flat
    # via copy_if_different (they are referenced by bare filename from cm_sdpa_common.hpp). A flat
    # copy of the experimental headers would not preserve the subdirectory and would fail the
    # include-verify step below.
    add_custom_command(
        OUTPUT "${OV_GPU_EXP_CM_INC}"
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${OV_GPU_EXP_CM_KERNEL_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OV_GPU_EXP_CM_HEADER_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FULL_EXP_CM_KERNELS} "${OV_GPU_EXP_CM_KERNEL_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${OV_GPU_EXP_DIR}/graph/impls/cm/include" "${OV_GPU_EXP_CM_HEADER_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OV_GPU_EXP_CM_UPSTREAM_HEADERS} "${OV_GPU_EXP_CM_HEADER_DIR}/"
        COMMAND ${Python3_EXECUTABLE} "${OV_GPU_EXP_DIR}/cm_codegen/verify_cm_includes.py"
                -in_kernels_dir "${OV_GPU_EXP_CM_KERNEL_DIR}"
                -in_headers_dir "${OV_GPU_EXP_CM_HEADER_DIR}"
        COMMAND ${Python3_EXECUTABLE} "${OV_GPU_CM_CODEGEN_SCRIPT}"
                -in_kernels_dir "${OV_GPU_EXP_CM_KERNEL_DIR}"
                -in_headers_dir "${OV_GPU_EXP_CM_HEADER_DIR}"
                -out_sources "${OV_GPU_EXP_CM_INC}"
                -out_headers "${OV_GPU_EXP_CODEGEN_DIR}/experimental_cm_headers.inc"
                -lang "cm"
        DEPENDS ${FULL_EXP_CM_KERNELS} ${FULL_EXP_CM_HEADERS} ${OV_GPU_EXP_CM_UPSTREAM_HEADERS}
                "${OV_GPU_EXP_DIR}/cm_codegen/verify_cm_includes.py" "${OV_GPU_CM_CODEGEN_SCRIPT}"
        COMMENT "Staging, verifying includes, and generating experimental CM kernel sources"
        VERBATIM)

    add_custom_target(ov_gpu_experimental_cm_codegen ALL DEPENDS "${OV_GPU_EXP_CM_INC}")

    # Experimental CM kernel template DB wraps the generated .inc (no upstream SourcesDB pollution).
    set(OV_GPU_EXP_CM_GRAPH_SOURCES
        ${OV_GPU_EXP_DIR}/cm_db/kernels_db.cpp
        ${OV_GPU_EXP_DIR}/cm_db/experimental_kernel_generator.cpp
        ${OV_GPU_EXP_DIR}/graph/impls/cm/block_sparse_attention/block_sparse_attention.cpp
        ${OV_GPU_EXP_DIR}/graph/impls/cm/draft_block_mask/draft_block_mask.cpp)
endif()

# -----------------------------------------------------------------------------
# Apply sources and include dirs to each graph/plugin twin.
# -----------------------------------------------------------------------------
macro(ov_gpu_add_experimental_sources_to_twin SUFFIX)
    target_sources(openvino_intel_gpu_graph${SUFFIX} PRIVATE ${OV_GPU_EXP_GRAPH_SOURCES})
    target_include_directories(openvino_intel_gpu_graph${SUFFIX} PRIVATE ${EXP_INCLUDE_DIRS})

    # The runtime library hosts execution_config.cpp, whose guarded requires_new_shape_infer() hook
    # references ov::op::experimental::{BlockSparseAttention, DraftBlockMask} under
    # #if defined(OPENVINO_EXPERIMENTAL). Give it the core experimental op headers so those guarded
    # includes resolve when the experimental path is enabled.
    target_include_directories(openvino_intel_gpu_runtime${SUFFIX} PRIVATE
        ${OV_CORE_EXP_INCLUDE}
        ${OV_CORE_EXP_SHAPE_INFER_INCLUDE})

    # The runtime library's execution_config.cpp hosts the guarded requires_new_shape_infer() hook
    # that references ov::op::experimental::{BlockSparseAttention, DraftBlockMask} under
    # #if defined(OPENVINO_EXPERIMENTAL). The plugin's op factories / hooks are guarded the same way
    # (Plugin::Plugin() calls register_primitives() only under that macro). Inject the define scoped
    # to exactly these targets (directory-level add_compile_definitions would leak it to every
    # downstream target in the intel_gpu subtree), so only the experimental twin compiles those
    # guarded branches.
    target_compile_definitions(openvino_intel_gpu_runtime${SUFFIX} PRIVATE OPENVINO_EXPERIMENTAL)
    target_compile_definitions(openvino_intel_gpu_plugin${SUFFIX} PRIVATE OPENVINO_EXPERIMENTAL)

    # The experimental sources must not rely on undefined preprocessor identifiers (e.g. an
    # accidentally-undefined OPENVINO_EXPERIMENTAL guard); promote -Wundef to an error so drift is
    # caught at compile time rather than silently changing which guarded branches get compiled. Use
    # set_property(SOURCE ... TARGET_DIRECTORY ...) so the flag is scoped to exactly the experimental
    # source files listed here, never leaking onto unrelated sources in the host plugin/graph targets.
    if(NOT MSVC)
        set_property(SOURCE ${OV_GPU_EXP_GRAPH_SOURCES}
            TARGET_DIRECTORY openvino_intel_gpu_graph${SUFFIX}
            APPEND PROPERTY COMPILE_OPTIONS -Wundef -Werror=undef)
        set_property(SOURCE ${OV_GPU_EXP_PLUGIN_SOURCES}
            TARGET_DIRECTORY openvino_intel_gpu_plugin${SUFFIX}
            APPEND PROPERTY COMPILE_OPTIONS -Wundef -Werror=undef)
        if(ENABLE_CM_FOR_GPU)
            set_property(SOURCE ${OV_GPU_EXP_CM_GRAPH_SOURCES}
                TARGET_DIRECTORY openvino_intel_gpu_graph${SUFFIX}
                APPEND PROPERTY COMPILE_OPTIONS -Wundef -Werror=undef)
        endif()
    endif()

    target_sources(openvino_intel_gpu_plugin${SUFFIX} PRIVATE ${OV_GPU_EXP_PLUGIN_SOURCES})
    target_include_directories(openvino_intel_gpu_plugin${SUFFIX} PRIVATE ${EXP_INCLUDE_DIRS})

    if(ENABLE_CM_FOR_GPU)
        target_sources(openvino_intel_gpu_graph${SUFFIX} PRIVATE ${OV_GPU_EXP_CM_GRAPH_SOURCES})
        target_include_directories(openvino_intel_gpu_graph${SUFFIX} PRIVATE ${OV_GPU_EXP_CODEGEN_DIR})
        add_dependencies(openvino_intel_gpu_graph${SUFFIX} ov_gpu_experimental_cm_codegen)
    endif()
endmacro()

ov_gpu_add_experimental_sources_to_twin("")
if(GPU_RT_TYPE STREQUAL "COMBINED")
    ov_gpu_add_experimental_sources_to_twin("_ocl")
endif()
