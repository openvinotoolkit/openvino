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

struct TokenTypePattern {
    std::string name;
    std::vector<int32_t> types;  // 0=text, 1=image
};

using PagedAttnTokenTypeParams = std::tuple<ov::element::Type_t,
                                            size_t,            //< head_size
                                            size_t,            //< head_num
                                            TokenTypePattern,  //< pattern
                                            std::string        //< Device name
                                            >;

class PagedAttentionTokenTypeTest : public testing::WithParamInterface<PagedAttnTokenTypeParams>,
                                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttnTokenTypeParams>& obj);

protected:
    void SetUp() override;

    void RunAndValidate(const std::vector<int32_t>& token_types,
                          const ov::Tensor& q_tensor,
                          const ov::Tensor& k_tensor,
                          const ov::Tensor& v_tensor,
                          const std::vector<float>& expected_output);

private:
    ov::Dimension::value_type headSize;
    ov::Dimension::value_type headNum;
};

}  // namespace test
}  // namespace ov