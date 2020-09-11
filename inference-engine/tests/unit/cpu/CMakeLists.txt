# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuUnitTests)

foreach (obj_lib IN LISTS mkldnn_plugin_object_libraries)
    list(APPEND mkldnn_object_files $<TARGET_OBJECTS:${obj_lib}>)
endforeach ()
list(APPEND MKLDNN_SRC_OBJ ${mkldnn_object_files})

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin
            ${IE_MAIN_SOURCE_DIR}/src/transformations/include
        OBJECT_FILES
            ${MKLDNN_SRC_OBJ}
        LINK_LIBRARIES
            unitTestUtils
            mkldnn
            inference_engine_transformations
        ADD_CPPLINT
        LABELS
            CPU
)