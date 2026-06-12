// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace test {

struct TestData {
    std::string name;
    std::vector<int32_t> tokenTypes;
    std::vector<float> qData;
    std::vector<float> kData;
    std::vector<float> vData;
    std::vector<float> expectedOutput;
    int slidingWindowSize;
};

class PagedAttentionTokenTypeTestData {
public:
    static std::vector<TestData> GetTestData();
};

}  // namespace test