# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(InferenceEngine)

add_subdirectory(thirdparty)
add_subdirectory(src)
add_subdirectory(ie_bridges/c)

add_subdirectory(samples)

# TODO: remove this
foreach(sample benchmark_app classification_sample_async hello_classification
                hello_nv12_input_classification hello_query_device hello_reshape_ssd
                ngraph_function_creation_sample object_detection_sample_ssd
                speech_sample style_transfer_sample hello_classification_c
                object_detection_sample_ssd_c hello_nv12_input_classification_c)
    if(TARGET ${sample})
        install(TARGETS ${sample}
                RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
    endif()
endforeach()

foreach(samples_library opencv_c_wrapper format_reader)
    if(TARGET ${samples_library})
        install(TARGETS ${samples_library}
                RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
                LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
    endif()
endforeach()

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
            DESTINATION samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.bat EXCLUDE
            PATTERN speech_libs_and_demos EXCLUDE
            PATTERN .clang-format EXCLUDE)
elseif(WIN32)
    install(DIRECTORY samples/
            DESTINATION samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.sh EXCLUDE
            PATTERN speech_libs_and_demos EXCLUDE
            PATTERN .clang-format EXCLUDE)
endif()

# install C samples

ie_cpack_add_component(c_samples DEPENDS core_c)

if(UNIX)
    install(PROGRAMS samples/build_samples.sh
            DESTINATION samples/c
            COMPONENT c_samples)
elseif(WIN32)
    install(PROGRAMS samples/build_samples_msvc.bat
            DESTINATION samples/c
            COMPONENT c_samples)
endif()

install(DIRECTORY ie_bridges/c/samples/
        DESTINATION samples/c
        COMPONENT c_samples
        PATTERN ie_bridges/c/samples/CMakeLists.txt EXCLUDE
        PATTERN ie_bridges/c/samples/.clang-format EXCLUDE)

install(FILES samples/CMakeLists.txt
        DESTINATION samples/c
        COMPONENT c_samples)
