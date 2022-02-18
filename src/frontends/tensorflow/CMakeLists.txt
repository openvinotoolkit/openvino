# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME tensorflow
                LINKABLE_FRONTEND
                SHUTDOWN_PROTOBUF
                SKIP_INSTALL
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::util openvino::runtime::dev)

#
# Temporary install steps
#

set(TARGET_NAME openvino_tensorflow_frontend)
set(TARGET_NAME_IRC openvino_tensorflow_fe)

set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME frontend::tensorflow)
export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
        APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

# install with original name for tests component
install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)

if(BUILD_SHARED_LIBS)
    function(ov_shared_library_name target_name library_name)
        set(LIB_PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}")
        set(LIB_SUFFIX "${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")

        set("${library_name}" "${LIB_PREFIX}${target_name}${LIB_SUFFIX}" PARENT_SCOPE)
    endfunction()

    function(ov_lib_file_name target_name library_name)
        set(LIB_PREFIX "${CMAKE_STATIC_LIBRARY_PREFIX}")
        set(LIB_SUFFIX "${IE_BUILD_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}")

        set("${library_name}" "${LIB_PREFIX}${target_name}${LIB_SUFFIX}" PARENT_SCOPE)
    endfunction()

    ov_shared_library_name(${TARGET_NAME_IRC} shared_library_name)

    # rename targets files to avoid auto-loading by FEM
    install(FILES $<TARGET_FILE:${TARGET_NAME}>
            DESTINATION ${IE_CPACK_RUNTIME_PATH}
            COMPONENT core
            RENAME ${shared_library_name})
    if(WIN32)
        ov_lib_file_name(${TARGET_NAME_IRC} lib_file_name)

        # need to install renamed .lib file as well
        install(FILES $<TARGET_LINKER_FILE:${TARGET_NAME}>
                DESTINATION ${IE_CPACK_LIBRARY_PATH}
                COMPONENT core
                RENAME ${lib_file_name})
    endif()
endif()

# install -dev part
install(DIRECTORY ${${TARGET_NAME}_INCLUDE_DIR}/openvino
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}/
        COMPONENT core_dev
        FILES_MATCHING PATTERN "*.hpp")
