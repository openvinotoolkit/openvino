# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(LIBRARY_OUTPUT_DIRECTORY_BIN ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

add_subdirectory(thirdparty/pybind11 EXCLUDE_FROM_ALL)
add_subdirectory(src/compatibility)

if(NGRAPH_UNIT_TEST_ENABLE)
    add_subdirectory(tests/mock/mock_py_ngraph_frontend)
    add_dependencies(_pyngraph mock_py_ngraph_frontend)
    set_target_properties(mock_py_ngraph_frontend PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            COMPILE_PDB_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            PDB_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN})

    add_subdirectory(tests/mock/pyngraph_fe_mock_api)
    add_dependencies(_pyngraph pybind_mock_frontend)
endif()
