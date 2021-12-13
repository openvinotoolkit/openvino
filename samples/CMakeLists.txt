# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(cpp)
add_subdirectory(c)

# used by tests_deprecated
if(TARGET format_reader)
    install(TARGETS format_reader
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
endif()

openvino_developer_export_targets(COMPONENT openvino_common TARGETS format_reader ie_samples_utils)

#
# Install
#

# install C++ samples

ie_cpack_add_component(${OV_COMP_CPP_SAMPLES} DEPENDS ${OV_COMP_CORE_DEV})

if(UNIX)
    install(DIRECTORY cpp/
            DESTINATION share/openvino/samples/cpp
            COMPONENT ${OV_COMP_CPP_SAMPLES}
            USE_SOURCE_PERMISSIONS
            PATTERN *.bat EXCLUDE
            PATTERN .clang-format EXCLUDE)
elseif(WIN32)
    install(DIRECTORY cpp/
            DESTINATION share/openvino/samples/cpp
            COMPONENT ${OV_COMP_CPP_SAMPLES}
            USE_SOURCE_PERMISSIONS
            PATTERN *.sh EXCLUDE
            PATTERN .clang-format EXCLUDE)
endif()

# install C samples

ie_cpack_add_component(${OV_COMP_C_SAMPLES} DEPENDS ${OV_COMP_CORE_C})

if(UNIX)
    install(PROGRAMS cpp/build_samples.sh
            DESTINATION share/openvino/samples/c
            COMPONENT ${OV_COMP_C_SAMPLES})
elseif(WIN32)
    install(PROGRAMS cpp/build_samples_msvc.bat
            DESTINATION share/openvino/samples/c
            COMPONENT ${OV_COMP_C_SAMPLES})
endif()

install(DIRECTORY c
        DESTINATION share/openvino/samples
        COMPONENT ${OV_COMP_C_SAMPLES}
        PATTERN c/CMakeLists.txt EXCLUDE
        PATTERN c/.clang-format EXCLUDE)

install(FILES cpp/CMakeLists.txt
        DESTINATION share/openvino/samples/c
        COMPONENT ${OV_COMP_C_SAMPLES})

# install Python samples

if(ENABLE_PYTHON)
    ie_cpack_add_component(${OV_COMP_PYTHON_SAMPLES})

    install(DIRECTORY python/
            DESTINATION share/openvino/samples/python
            USE_SOURCE_PERMISSIONS
            COMPONENT ${OV_COMP_PYTHON_SAMPLES})
endif()
