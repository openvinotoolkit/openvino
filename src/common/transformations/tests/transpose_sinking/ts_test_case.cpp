// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ts_test_case.hpp"

using namespace transpose_sinking::testing;

::testing::internal::CartesianProductHolder<::testing::internal::ParamGenerator <
                                            unsigned long>, ::testing::internal::ParamGenerator<unsigned long>, ::testing::internal::ValueArray <TestCase>>
transpose_sinking::testing::wrapper(const TestCase &test_case) {
    OPENVINO_ASSERT(test_case.model.main_op.size() == test_case.model_ref.main_op.size(),
                    "The number of main op (testing op) creator have to be the same for the testing model and for"
                    "the reference model.");
    return ::testing::Combine(::testing::Range<size_t>(0, test_case.num_main_ops.size()),
                              ::testing::Range<size_t>(0, test_case.model.main_op.size()),
                              ::testing::Values(test_case));
}

std::string TSTestFixture::get_test_name(const ::testing::TestParamInfo<TestParams> &obj) {
    size_t num_main_ops_idx;
    size_t main_op_idx;
    TestCase test_case;
    std::tie(num_main_ops_idx, main_op_idx, test_case) = obj.param;

    std::ostringstream test_name;
    test_name << "Factory=" << test_case.model.main_op[main_op_idx]->getTypeName() << "/";
    test_name << "NumOps=" << test_case.num_main_ops[num_main_ops_idx] << "/";
    test_name << "Transformation=" << test_case.transformation->getTypeName() << "/";
    return test_name.str();
}
