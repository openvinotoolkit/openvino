# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED REF_LIBRARY)
    message(FATAL_ERROR "Reference library path is not defined")
endif()

if(NOT EXISTS "${REF_LIBRARY}")
    message(FATAL_ERROR "Reference library does not exist")
endif()

if(NOT DEFINED ACTUAL_LIBRARY)
    message(FATAL_ERROR "Actual library path is not defined")
endif()

if(NOT EXISTS "${ACTUAL_LIBRARY}")
    message(FATAL_ERROR "Actual library does not exist")
endif()

file(SIZE "${REF_LIBRARY}" ref_size)
file(SIZE "${ACTUAL_LIBRARY}" compressed_size)

message("Reference runtime size: ${ref_size} bytes")
message("Custom runtime size: ${compressed_size} bytes")

math(EXPR value "${compressed_size} * 100 / ${ref_size}" OUTPUT_FORMAT DECIMAL)
message("Ratio: ${value} %")
