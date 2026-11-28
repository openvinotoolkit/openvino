// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using PagedAttnRelativeBiasParams = std::tuple<ov::element::Type_t,  //< inference precision
                                               size_t,               //< head_size
                                               size_t,               //< head_num
                                               size_t,      //< relative bias distances D (table has D + 1 entries)
                                               size_t,      //< sliding_window_size
                                               size_t,      //< batch_size
                                               size_t,      //< sequence_length
                                               std::string  //< Device name
                                               >;

class PagedAttentionRelativeBiasTest : public testing::WithParamInterface<PagedAttnRelativeBiasParams>,
                                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttnRelativeBiasParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

}  // namespace test
}  // namespace ov