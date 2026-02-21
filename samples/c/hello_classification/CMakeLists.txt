# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_sample(NAME hello_classification_c
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.c;${CMAKE_CURRENT_SOURCE_DIR}/infer_result_util.c;${CMAKE_CURRENT_SOURCE_DIR}/ov_dynamic_loader.c"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/infer_result_util.h;${CMAKE_CURRENT_SOURCE_DIR}/ov_dynamic_loader.h"
              DEPENDENCIES opencv_c_wrapper ie_samples_util_c)
get_target_property(_libs hello_classification_c INTERFACE_LINK_LIBRARIES)
if(_libs)
    list(REMOVE_ITEM _libs openvino::runtime::c)
    set_target_properties(hello_classification_c PROPERTIES INTERFACE_LINK_LIBRARIES "${_libs}")
endif()
