# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT CPACK_GENERATOR STREQUAL "DEB")
    set(linkable_frontend LINKABLE_FRONTEND)
endif()

ov_add_frontend(NAME tensorflow
                ${linkable_frontend}
                SHUTDOWN_PROTOBUF
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::util openvino::runtime::dev)

# give a different name during installation to OpenVINO package
set(target_name "${FRONTEND_NAME_PREFIX}tensorflow${FRONTEND_NAME_SUFFIX}")
set_target_properties(${target_name} PROPERTIES OUTPUT_NAME openvino_tensorflow_fe)

if(NOT TARGET openvino::frontend::tensorflow)
    add_library(openvino::frontend::tensorflow ALIAS ${target_name})
endif()

if(ENABLE_TESTS)
    function(ov_frontend_get_file_name target_name library_name)
        set(LIB_PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}")

        set(LIB_SUFFIX "${IE_BUILD_POSTFIX}")
        if(APPLE)
            set(LIB_SUFFIX "${LIB_SUFFIX}${OpenVINO_VERSION_SUFFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")
        else()
            set(LIB_SUFFIX "${LIB_SUFFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}${OpenVINO_VERSION_SUFFIX}")
        endif()

        set("${library_name}" "${LIB_PREFIX}${target_name}${LIB_SUFFIX}" PARENT_SCOPE)
    endfunction()

    ov_frontend_get_file_name(${target_name} output_name)

    # install with original name for tests component
    install(FILES $<TARGET_FILE:${target_name}>
            DESTINATION tests
            COMPONENT tests
            RENAME ${output_name}
            EXCLUDE_FROM_ALL)

    install(TARGETS ${target_name}
            LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

    add_subdirectory(tests)
endif()
