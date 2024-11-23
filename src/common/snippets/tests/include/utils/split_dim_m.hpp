// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>

namespace ov {
namespace test {
namespace snippets {

struct InputData {
    size_t cur_batch;
    size_t cur_m;
    size_t concurrency;
};

struct ReferenceData {
    bool is_split;
    size_t batch_m;
    size_t kernel_m;
};

struct SplitDimensionMParams {
    InputData input;
    ReferenceData reference;
};

class SplitDimensionMTest : public testing::TestWithParam<SplitDimensionMParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SplitDimensionMParams> obj);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
