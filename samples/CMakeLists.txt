# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

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

# install C samples

ie_cpack_add_component(c_samples DEPENDS core_c)

if(UNIX)
    install(PROGRAMS ${IE_MAIN_SOURCE_DIR}/samples/build_samples.sh
            DESTINATION samples/c
            COMPONENT c_samples)
elseif(WIN32)
    install(PROGRAMS ${IE_MAIN_SOURCE_DIR}/samples/build_samples_msvc.bat
            DESTINATION samples/c
            COMPONENT c_samples)
endif()

install(DIRECTORY c
        DESTINATION samples
        COMPONENT c_samples
        PATTERN c/CMakeLists.txt EXCLUDE
        PATTERN c/.clang-format EXCLUDE)

install(FILES ${IE_MAIN_SOURCE_DIR}/samples/CMakeLists.txt
        DESTINATION samples/c
        COMPONENT c_samples)

# install Python samples

ie_cpack_add_component(python_samples)

install(DIRECTORY python/
        DESTINATION samples/python
        USE_SOURCE_PERMISSIONS
        COMPONENT python_samples)
