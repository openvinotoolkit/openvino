// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

struct TestData {
    std::string name;
    std::vector<int32_t> tokenTypes;
    std::vector<float> qData;
    std::vector<float> kData;
    std::vector<float> vData;
    std::vector<float> expectedOutput;
};

using PagedAttnTokenTypeParams = std::tuple<ov::element::Type_t,
                                            size_t,      //< head_size
                                            size_t,      //< head_num
                                            int32_t,     //< sliding_window_size
                                            TestData,    //< pattern
                                            std::string, //< Device name
                                            bool         //< use_flash_attn_v2
                                            >;

class PagedAttentionTokenTypeTest : public testing::WithParamInterface<PagedAttnTokenTypeParams>,
                                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttnTokenTypeParams>& obj);
    static std::vector<TestData> GetTestDataForHeadSize32HeadNum1();
    static std::vector<TestData> GetTestDataForHeadSize32HeadNum1SlidingWindowSize5();
    void run() override;

protected:
    void SetUp() override;
    void TearDown() override;
    void RunAndValidate();
};

}  // namespace test
}  // namespace ov