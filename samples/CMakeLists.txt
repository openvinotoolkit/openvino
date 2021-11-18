# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(cpp)

# TODO: remove this
foreach(sample benchmark_app classification_sample_async hello_classification
                hello_nv12_input_classification hello_query_device hello_reshape_ssd
                ngraph_function_creation_sample object_detection_sample_ssd
                speech_sample style_transfer_sample)
    if(TARGET ${sample})
        install(TARGETS ${sample}
                RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
    endif()
endforeach()

if(TARGET format_reader)
    install(TARGETS format_reader
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
endif()

openvino_developer_export_targets(COMPONENT openvino_common TARGETS format_reader ie_samples_utils)

add_subdirectory(c)

# TODO: remove this
foreach(sample hello_classification_c
        object_detection_sample_ssd_c hello_nv12_input_classification_c)
    if(TARGET ${sample})
        install(TARGETS ${sample}
                RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
    endif()
endforeach()

if(TARGET opencv_c_wrapper)
    install(TARGETS opencv_c_wrapper
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
endif()
#
# Install
#

# install C++ samples

ie_cpack_add_component(cpp_samples DEPENDS cpp_samples_deps core)

if(UNIX)
    install(DIRECTORY cpp/
            DESTINATION samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.bat EXCLUDE
            PATTERN .clang-format EXCLUDE)
elseif(WIN32)
    install(DIRECTORY cpp/
            DESTINATION samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.sh EXCLUDE
            PATTERN .clang-format EXCLUDE)
endif()

# install C samples

ie_cpack_add_component(c_samples DEPENDS core_c)

if(UNIX)
    install(PROGRAMS cpp/build_samples.sh
            DESTINATION samples/c
            COMPONENT c_samples)
elseif(WIN32)
    install(PROGRAMS cpp/build_samples_msvc.bat
            DESTINATION samples/c
            COMPONENT c_samples)
endif()

install(DIRECTORY c
        DESTINATION samples
        COMPONENT c_samples
        PATTERN c/CMakeLists.txt EXCLUDE
        PATTERN c/.clang-format EXCLUDE)

install(FILES cpp/CMakeLists.txt
        DESTINATION samples/c
        COMPONENT c_samples)

# install Python samples

ie_cpack_add_component(python_samples)

install(DIRECTORY python/
        DESTINATION samples/python
        USE_SOURCE_PERMISSIONS
        COMPONENT python_samples)
