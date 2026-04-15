// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>
#include <optional>

namespace ov {
namespace test {
namespace snippets {

struct InputData {
    std::optional<size_t> cur_batch;
    size_t cur_m;
    size_t concurrency;
};

struct ReferenceData {
    bool is_split;
    size_t batch_m;
    size_t kernel_m;
};

struct MHAParallelWASplitParams {
    InputData input;
    ReferenceData reference;
};

class MHAParallelWASplitTest : public testing::TestWithParam<MHAParallelWASplitParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MHAParallelWASplitParams> obj);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
