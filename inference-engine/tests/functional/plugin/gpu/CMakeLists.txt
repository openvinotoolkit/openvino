# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME gpuFuncTests)

addIeTargetTest(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CLDNN__IOCL_ICD_INCDIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            clDNNPlugin
        LINK_LIBRARIES
            funcSharedTests
            ${CLDNN__IOCL_ICD_LIBPATH}
        ADD_CPPLINT
        LABELS
            GPU
)

if(LIBVA_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LIBVA)
    target_include_directories(${TARGET_NAME} PRIVATE ${LIBVA_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBVA_LINK_LIBRARIES})
endif()
