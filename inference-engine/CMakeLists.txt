# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(InferenceEngine)

add_subdirectory(thirdparty)
add_subdirectory(src)
add_subdirectory(ie_bridges/c)

if(ENABLE_PYTHON)
    add_subdirectory(ie_bridges/python)
endif()

add_subdirectory(tools)
add_subdirectory(samples)

openvino_developer_export_targets(COMPONENT openvino_common TARGETS format_reader ie_samples_utils)

if(ENABLE_TESTS)
    add_subdirectory(tests_deprecated)
    add_subdirectory(tests)
endif()

#
# Install
#

# install C++ samples

ie_cpack_add_component(cpp_samples DEPENDS cpp_samples_deps core)

if(UNIX)
    install(DIRECTORY samples/
            DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.bat EXCLUDE
            PATTERN speech_libs_and_demos EXCLUDE
            PATTERN .clang-format EXCLUDE)
elseif(WIN32)
    install(DIRECTORY samples/
            DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.sh EXCLUDE
            PATTERN speech_libs_and_demos EXCLUDE
            PATTERN .clang-format EXCLUDE)
endif()

install(TARGETS format_reader
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)

if(TARGET benchmark_app)
    install(TARGETS benchmark_app
            RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
endif()

# install C samples

ie_cpack_add_component(c_samples DEPENDS core_c)

if(UNIX)
    install(PROGRAMS samples/build_samples.sh
            DESTINATION ${IE_CPACK_IE_DIR}/samples/c
            COMPONENT c_samples)
elseif(WIN32)
    install(PROGRAMS samples/build_samples_msvc.bat
            DESTINATION ${IE_CPACK_IE_DIR}/samples/c
            COMPONENT c_samples)
endif()

install(DIRECTORY ie_bridges/c/samples/
        DESTINATION ${IE_CPACK_IE_DIR}/samples/c
        COMPONENT c_samples
        PATTERN ie_bridges/c/samples/CMakeLists.txt EXCLUDE
        PATTERN ie_bridges/c/samples/.clang-format EXCLUDE)

install(FILES samples/CMakeLists.txt
        DESTINATION ${IE_CPACK_IE_DIR}/samples/c
        COMPONENT c_samples)

# install speech demo files

if(SPEECH_LIBS_AND_DEMOS)
    ie_cpack_add_component(speech_demo_files)

    install(DIRECTORY ${TEMP}/deployment_tools
                      ${TEMP}/data_processing
            DESTINATION .
            USE_SOURCE_PERMISSIONS
            COMPONENT speech_demo_files)
endif()
