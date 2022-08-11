# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME tensorflow
                LINKABLE_FRONTEND
                SHUTDOWN_PROTOBUF
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::util openvino::runtime::dev)

# give a different name during installation to OpenVINO package
set(target_name "${FRONTEND_NAME_PREFIX}tensorflow${FRONTEND_NAME_SUFFIX}")
set_target_properties(${target_name} PROPERTIES OUTPUT_NAME openvino_tensorflow_fe)

function(ov_frontend_get_file_name target_name library_name)
    set(LIB_PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}")
    set(LIB_SUFFIX "${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}${OpenVINO_VERSION_SUFFIX}")

    set("${library_name}" "${LIB_PREFIX}${target_name}${LIB_SUFFIX}" PARENT_SCOPE)
endfunction()

ov_frontend_get_file_name(${target_name} output_name)

# install with original name for tests component
install(FILES $<TARGET_FILE:${target_name}>
        DESTINATION tests
        COMPONENT tests
        RENAME ${output_name}
        EXCLUDE_FROM_ALL)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
