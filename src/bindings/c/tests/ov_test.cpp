// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

#include "test_model_repo.hpp"

std::string xml_std = TestDataHelpers::generate_model_path("test_model", "test_model_fp32.xml");
std::string bin_std = TestDataHelpers::generate_model_path("test_model", "test_model_fp32.bin");

const char* xml = xml_std.c_str();
const char* bin = bin_std.c_str();

std::map<ov_element_type_e, size_t> element_type_size_map = {{ov_element_type_e::BOOLEAN, 8},
                                                             {ov_element_type_e::BF16, 16},
                                                             {ov_element_type_e::F16, 16},
                                                             {ov_element_type_e::F32, 32},
                                                             {ov_element_type_e::F64, 64},
                                                             {ov_element_type_e::I4, 4},
                                                             {ov_element_type_e::I8, 8},
                                                             {ov_element_type_e::I16, 16},
                                                             {ov_element_type_e::I32, 32},
                                                             {ov_element_type_e::I64, 64},
                                                             {ov_element_type_e::U1, 1},
                                                             {ov_element_type_e::U4, 4},
                                                             {ov_element_type_e::U8, 8},
                                                             {ov_element_type_e::U16, 16},
                                                             {ov_element_type_e::U32, 32},
                                                             {ov_element_type_e::U64, 64}};
