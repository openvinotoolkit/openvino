// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ts_test_case.hpp"

using namespace transpose_sinking::testing;

std::string TSTestFixture::get_test_name(const ::testing::TestParamInfo<TestParams>& obj) {
    const auto& [num_main_ops_idx, main_op_idx, test_case] = obj.param;

    std::ostringstream test_name;
    test_name << "Factory=" << test_case.model.main_op[main_op_idx]->getTypeName() << "/";
    test_name << "NumOps=" << test_case.num_main_ops[num_main_ops_idx] << "/";
    test_name << "Transformation=" << test_case.transformation->getTypeName() << "/";
    return test_name.str();
}
