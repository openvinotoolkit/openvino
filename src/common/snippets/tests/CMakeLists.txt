# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_snippets_func_tests)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/linear_ir.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/loop_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/buffer_allocation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/compute_buffer_allocation_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/extracted_loop_invariants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/init_loops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/insert_load_store.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/insert_specific_iterations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/loop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/move_scalar_to_consumer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/optimize_domain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/split_loops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/canonicalization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/collapse_subgraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/fake_quantize_decomposition_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/fuse_transpose_brgemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/gated_mlp_tokenization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/gn_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/gn_tokenization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/mha_tokenization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/mlp_seq_tokenization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/online_softmax_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/precision_propagation_convert_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/precision_propagation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/precision_propagation_get_precisions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/softmax_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/softmax_reshape_elimination.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime_configurator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/broadcast_merge_dim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/split_dim_m.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/precomp.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/lir_comparator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/lir_test_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/lowered/pass/buffer_allocation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/lowered/pass/optimize_domain.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/lowering_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pass/canonicalization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pass/collapse_subgraph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pass/fuse_transpose_brgemm.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pass/gated_mlp_tokenization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pass/gn_decomposition.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pass/gn_tokenization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pass/mha_tokenization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pass/mlp_seq_tokenization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pass/online_softmax_decomposition.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pass/precision_propagation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pass/softmax_decomposition.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/broadcast_dim_merge.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/split_dim_m.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils.hpp
)

source_group("src"     FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<TARGET_PROPERTY:openvino::snippets,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime::dev
    common_test_utils
    ov_snippets_models
    snippets_test_utils
)

if(EMSCRIPTEN)
    set(JS_BIN_NAME "${TARGET_NAME}.js")
    set(JS_APP_NAME "${TARGET_NAME}_js.js")
    set(JS_TEST_APP "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${JS_APP_NAME}")
    file(WRITE   ${JS_TEST_APP} "// Copyright (C) 2018-2026 Intel Corporation\n")
    file(APPEND  ${JS_TEST_APP} "// SPDX-License-Identifier: Apache-2.0\n")
    file(APPEND  ${JS_TEST_APP} "//\n")
    file(APPEND  ${JS_TEST_APP} "// JS test app\n")
    file(APPEND  ${JS_TEST_APP} "const createModule = require(\"./${JS_BIN_NAME}\");\n")
    file(APPEND  ${JS_TEST_APP} "createModule().then(function(Module) {});\n")
    file(APPEND  ${JS_TEST_APP} " ")
    add_test(NAME ${TARGET_NAME} COMMAND node ${JS_TEST_APP})
else()
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endif()
set_property(TEST ${TARGET_NAME} PROPERTY LABELS OV UNIT SNIPPETS)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH_HEADER "src/precomp.hpp"
)

add_library(snippets_test_utils STATIC ${CMAKE_CURRENT_SOURCE_DIR}/include/lowering_utils.hpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/include/lir_test_utils.hpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/include/utils.hpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/include/lir_comparator.hpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/include/lowered/pass/buffer_allocation.hpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/src/lowering_utils.cpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/src/lir_test_utils.cpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/src/lir_comparator.cpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/src/lowered/pass/buffer_allocation.cpp)
target_include_directories(snippets_test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(snippets_test_utils PRIVATE common_test_utils ov_snippets_models)
