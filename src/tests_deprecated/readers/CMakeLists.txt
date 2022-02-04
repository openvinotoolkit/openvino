# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_IR_V7_READER)
    add_subdirectory(ir_reader_v7)
    add_dependencies(ov_runtime_libraries inference_engine_ir_v7_reader)
endif()
