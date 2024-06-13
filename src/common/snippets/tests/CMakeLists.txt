# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_snippets_func_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            $<TARGET_PROPERTY:openvino::snippets,INTERFACE_INCLUDE_DIRECTORIES>
        LINK_LIBRARIES
            openvino::runtime::dev
            common_test_utils
            ov_snippets_models
        ADD_CPPLINT
        LABELS
            OV UNIT SNIPPETS
)

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)

add_library(snippets_test_utils STATIC ${CMAKE_CURRENT_SOURCE_DIR}/include/lowering_utils.hpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/include/lir_test_utils.hpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/include/lir_comparator.hpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/src/lowering_utils.cpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/src/lir_test_utils.cpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/src/lir_comparator.cpp)
target_include_directories(snippets_test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(snippets_test_utils PRIVATE common_test_utils ov_snippets_models)
