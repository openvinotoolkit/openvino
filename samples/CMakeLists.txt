# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(cpp)
add_subdirectory(c)

# used by tests_deprecated
if(TARGET format_reader)
    install(TARGETS format_reader
            RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT tests EXCLUDE_FROM_ALL)
endif()

openvino_developer_export_targets(COMPONENT openvino_common TARGETS format_reader ie_samples_utils)

#
# Install
#

# install C++ samples

ie_cpack_add_component(${OV_CPACK_COMP_CPP_SAMPLES} DEPENDS ${OV_CPACK_COMP_CORE_DEV})

if(UNIX)
    install(DIRECTORY cpp/
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
            USE_SOURCE_PERMISSIONS
            PATTERN *.bat EXCLUDE
            PATTERN .clang-format EXCLUDE)
elseif(WIN32)
    install(DIRECTORY cpp/
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
            USE_SOURCE_PERMISSIONS
            PATTERN *.sh EXCLUDE
            PATTERN .clang-format EXCLUDE)
endif()

# install C samples

ie_cpack_add_component(${OV_CPACK_COMP_C_SAMPLES} DEPENDS ${OV_CPACK_COMP_CORE_C})

if(UNIX)
    install(PROGRAMS cpp/build_samples.sh
            DESTINATION ${OV_CPACK_SAMPLESDIR}/c
            COMPONENT ${OV_CPACK_COMP_C_SAMPLES})
elseif(WIN32)
    install(PROGRAMS cpp/build_samples_msvc.bat
            DESTINATION ${OV_CPACK_SAMPLESDIR}/c
            COMPONENT ${OV_CPACK_COMP_C_SAMPLES})
endif()

install(DIRECTORY c
        DESTINATION ${OV_CPACK_SAMPLESDIR}
        COMPONENT ${OV_CPACK_COMP_C_SAMPLES}
        PATTERN c/CMakeLists.txt EXCLUDE
        PATTERN c/.clang-format EXCLUDE)

install(FILES cpp/CMakeLists.txt
        DESTINATION ${OV_CPACK_SAMPLESDIR}/c
        COMPONENT ${OV_CPACK_COMP_C_SAMPLES})

# install Python samples

ie_cpack_add_component(${OV_CPACK_COMP_PYTHON_SAMPLES})

install(DIRECTORY python/
        DESTINATION ${OV_CPACK_SAMPLESDIR}/python
        USE_SOURCE_PERMISSIONS
        COMPONENT ${OV_CPACK_COMP_PYTHON_SAMPLES})
