# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(PYTHONLIBS_VERSION_STRING MATCHES "^([0-9]+)\.([0-9]+).*")
    set(PYTHON_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(PYTHON_VERSION_MINOR ${CMAKE_MATCH_2})
    set(PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
else()
    message(FATAL_ERROR "Failed to extract python major.minor from ${PYTHONLIBS_VERSION_STRING}")
endif()

if(ENABLE_CONDA_FOLDER)
    set(PYTHON_COMPONENT conda_${PYTHON_VERSION})
else()
    set(PYTHON_COMPONENT ${PYTHON_VERSION})
endif()

# install Python samples

ie_cpack_add_component(python_samples)

install(DIRECTORY samples/
        DESTINATION samples/python
        USE_SOURCE_PERMISSIONS
        COMPONENT python_samples)

ie_cpack(${PYTHON_COMPONENT} python_samples)
