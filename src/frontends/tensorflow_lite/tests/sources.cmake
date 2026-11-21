# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_TENSORFLOW_LITE_TESTS_SRC
    ${CMAKE_CURRENT_LIST_DIR}/basic_api.cpp
    ${CMAKE_CURRENT_LIST_DIR}/conversion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/convert_model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/convert_sparse_incomplete.cpp
    ${CMAKE_CURRENT_LIST_DIR}/convert_tricky_models.cpp
    ${CMAKE_CURRENT_LIST_DIR}/convert_unsupported.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library_extension.cpp
    ${CMAKE_CURRENT_LIST_DIR}/op_extension.cpp
    ${CMAKE_CURRENT_LIST_DIR}/quantization.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sparsity_info_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/telemetry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tf_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tf_utils.hpp
)

set(TENSORFLOW_GEN_SCRIPTS
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_2in_2out.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_bad_buffer_size.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_bad_header.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_densify.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_dequantize.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_malformed_indices.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_malformed_tensor_name.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_oob_builtin_opcode.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_oob_quantized_dimension.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_qdq.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_quantize.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_sparse_incomplete.py
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_scripts/generate_sparse_oob.py
)
set(TENSORFLOW_ALL_SCRIPTS
    ${TENSORFLOW_GEN_SCRIPTS}
    ${CMAKE_CURRENT_LIST_DIR}/test_models/gen_wrapper.py
)
