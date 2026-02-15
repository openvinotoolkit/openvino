# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(cpp)
add_subdirectory(c)

ov_developer_package_export_targets(TARGET format_reader)
ov_developer_package_export_targets(TARGET ie_samples_utils)

#
# Install
#

# install C++ samples

ov_cpack_add_component(${OV_CPACK_COMP_CPP_SAMPLES}
                       HIDDEN
                       DEPENDS ${OV_CPACK_COMP_CORE_DEV})

if(UNIX)
    install(DIRECTORY cpp/
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
            ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL}
            PATTERN *.bat EXCLUDE
            PATTERN *.ps1 EXCLUDE
            PATTERN *.sh EXCLUDE
            PATTERN .clang-format EXCLUDE)

    install(PROGRAMS cpp/build_samples.sh
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
            ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL})
elseif(WIN32)
    install(DIRECTORY cpp/
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
            ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL}
            PATTERN *.sh EXCLUDE
            PATTERN .clang-format EXCLUDE)
endif()

# install C samples

ov_cpack_add_component(${OV_CPACK_COMP_C_SAMPLES}
                       HIDDEN
                       DEPENDS ${OV_CPACK_COMP_CORE_C_DEV})

if(UNIX)
    install(PROGRAMS cpp/build_samples.sh
            DESTINATION ${OV_CPACK_SAMPLESDIR}/c
            COMPONENT ${OV_CPACK_COMP_C_SAMPLES}
            ${OV_CPACK_COMP_C_SAMPLES_EXCLUDE_ALL})
elseif(WIN32)
    install(PROGRAMS cpp/build_samples_msvc.bat cpp/build_samples.ps1
            DESTINATION ${OV_CPACK_SAMPLESDIR}/c
            COMPONENT ${OV_CPACK_COMP_C_SAMPLES}
            ${OV_CPACK_COMP_C_SAMPLES_EXCLUDE_ALL})
endif()

install(DIRECTORY c
        DESTINATION ${OV_CPACK_SAMPLESDIR}
        COMPONENT ${OV_CPACK_COMP_C_SAMPLES}
        ${OV_CPACK_COMP_C_SAMPLES_EXCLUDE_ALL}
        PATTERN c/CMakeLists.txt EXCLUDE
        PATTERN c/.clang-format EXCLUDE)

install(FILES cpp/CMakeLists.txt
        DESTINATION ${OV_CPACK_SAMPLESDIR}/c
        COMPONENT ${OV_CPACK_COMP_C_SAMPLES}
        ${OV_CPACK_COMP_C_SAMPLES_EXCLUDE_ALL})

# install Python samples

ov_cpack_add_component(${OV_CPACK_COMP_PYTHON_SAMPLES} HIDDEN)

install(DIRECTORY python/
        DESTINATION ${OV_CPACK_SAMPLESDIR}/python
        COMPONENT ${OV_CPACK_COMP_PYTHON_SAMPLES}
        ${OV_CPACK_COMP_PYTHON_SAMPLES_EXCLUDE_ALL})
