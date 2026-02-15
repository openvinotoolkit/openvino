# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var IN ITEMS BUILD_DIR OUTPUT_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "Internal error: ${var} is not defined")
    endif()
endforeach()

file(GLOB_RECURSE PDB_FILES "${BUILD_DIR}/*.pdb")

message("Found PDBs for oneDNN GPU: ")
foreach(pdb IN LISTS PDB_FILES)
    message("${pdb}")
endforeach()

file(COPY ${PDB_FILES} DESTINATION ${OUTPUT_DIR})
