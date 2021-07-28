// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_irs.hpp"
#include <gtest/gtest.h>
#include "../gna_matcher.hpp"

using namespace GNATestIRs::Permute;

typedef struct { Permute3dimCaseParam test_param; bool supported; } Permute3dimTestParam;

class GNAPermute3dTest : public GNATest<::testing::TestWithParam<Permute3dimTestParam>> {

};

static std::string getPermute3dTestName(testing::TestParamInfo<Permute3dimTestParam> obj) {
    std::string test_name = "order";
    for (int n = 0; n < 3; n++) {
        test_name += "_" + std::to_string(obj.param.test_param.order[n]);
    }
    test_name += "_dim";
    for (int n = 0; n < 3; n++) {
        test_name += "_" + std::to_string(obj.param.test_param.dim[n]);
    }
    return test_name;
}

TEST_P(GNAPermute3dTest, Permute3dim) {
    auto test_param = GetParam().test_param;
    auto supported = GetParam().supported;

    std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<float> weights = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                  0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
    std::vector<float> expected_result(2);
    for (int res_index = 0; res_index < expected_result.size(); res_index++) {
        for (int n = 0; n < input_data.size(); n++) {
        expected_result[res_index] += input_data[n] * weights[res_index * input_data.size() + n];
        }
    }

    auto& test_instance =
            assert_that().onInferModel(Permute3dimModel_v6(test_param))
                    .inNotCompactMode()
                    .withWeigthsPattern(std::move(weights))
                    .gna()
                    .propagate_forward()
                    .onCPU()
                    .called_with_input(input_data);
    if (supported) {
        test_instance.equals_to(expected_result);
    } else {
        test_instance.throws();
    }
}

const Permute3dimTestParam gna_permute3d_test_params[] = {
        {{{1, 0, 2}, {1, 2, 4}}, true},
        {{{1, 0, 2}, {2, 1, 4}}, true},
        {{{1, 0, 2}, {1, 4, 2}}, true},
        {{{1, 0, 2}, {4, 1, 2}}, true},
        {{{1, 0, 2}, {1, 8, 1}}, true},
        {{{1, 0, 2}, {8, 1, 1}}, true},
        {{{1, 0, 2}, {4, 2, 1}}, false},
        {{{1, 0, 2}, {2, 4, 1}}, false},
        {{{1, 2, 0}, {1, 2, 4}}, true},
        {{{0, 1, 2}, {1, 2, 4}}, true},
        {{{0, 2, 1}, {2, 1, 4}}, true},
        {{{2, 0, 1}, {1, 2, 4}}, false},
        {{{2, 1, 0}, {2, 1, 4}}, false}
};

INSTANTIATE_TEST_SUITE_P(GNALayerTests, GNAPermute3dTest,
        ::testing::ValuesIn(gna_permute3d_test_params), getPermute3dTestName);
