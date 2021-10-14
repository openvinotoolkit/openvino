# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_SAMPLES)
    add_subdirectory(C)
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

install(DIRECTORY C
        DESTINATION samples/c
        COMPONENT c_samples
        PATTERN C/CMakeLists.txt EXCLUDE
        PATTERN C/.clang-format EXCLUDE)

install(FILES ${IE_MAIN_SOURCE_DIR}/samples/CMakeLists.txt
        DESTINATION samples/c
        COMPONENT c_samples)

# install Python samples

ie_cpack_add_component(python_samples)

install(DIRECTORY python/
        DESTINATION samples/python
        USE_SOURCE_PERMISSIONS
        COMPONENT python_samples)
