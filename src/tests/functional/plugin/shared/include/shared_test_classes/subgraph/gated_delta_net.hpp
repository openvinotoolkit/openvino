// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using gated_delta_net_params = std::tuple<int32_t,            // B
                                          int32_t,            // T
                                          int32_t,            // qk_head_nums
                                          int32_t,            // v_head_nums
                                          int32_t,            // qk_head_size
                                          int32_t,            // v_head_size
                                          ov::element::Type,  // infer_precision
                                          std::string         // device
                                          >;

class GatedDeltaNet : public testing::WithParamInterface<gated_delta_net_params>, public ov::test::SubgraphBaseTest {
private:
    std::shared_ptr<ov::Model> buildLoopedGDN(int32_t batch,
                                              int32_t seq_len,
                                              int32_t qk_head_num,
                                              int32_t v_head_num,
                                              int32_t qk_head_size,
                                              int32_t v_head_size,
                                              ov::element::Type dtype);

public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    static std::string getTestCaseName(const testing::TestParamInfo<gated_delta_net_params>& obj);

protected:
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
