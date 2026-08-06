// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

using selective_ssm_params = std::tuple<int32_t,            // B
                                        int32_t,            // T
                                        int32_t,            // H
                                        int32_t,            // G
                                        int32_t,            // P
                                        int32_t,            // N
                                        ov::element::Type,  // infer_precision
                                        std::string         // device
                                        >;

class SelectiveSSM : public testing::WithParamInterface<selective_ssm_params>, public ov::test::SubgraphBaseTest {
private:
    std::shared_ptr<ov::Model> buildLoopedSelectiveSSM(int32_t num_heads,
                                                       int32_t num_groups,
                                                       int32_t head_dim,
                                                       int32_t state_size,
                                                       ov::element::Type dtype);

public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    static std::string getTestCaseName(const testing::TestParamInfo<selective_ssm_params>& obj);

protected:
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void SetUp() override;
};

}  // namespace ov::test
