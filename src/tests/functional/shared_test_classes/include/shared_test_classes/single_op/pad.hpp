// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<int64_t>,      // padsBegin
        std::vector<int64_t>,      // padsEnd
        float,                     // argPadValue
        ov::test::utils::PadMode,  // padMode
        ov::element::Type,         // Net precision
        std::vector<InputShape>,   // Input shapes
        std::string                // Target device name
> padLayerTestParamsSet;

class PadLayerTest : public testing::WithParamInterface<padLayerTestParamsSet>,
                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<padLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
    virtual std::shared_ptr<ov::Node> create_pad_op(const ngraph::Output<ov::Node>& data,
                                      const std::vector<int64_t>& padsBegin,
                                      const std::vector<int64_t>& padsEnd,
                                      float argPadValue,
                                      ov::test::utils::PadMode padMode) const;
};

class PadLayerTest12 : public PadLayerTest {
protected:
        std::shared_ptr<ov::Node> create_pad_op(const ngraph::Output<ov::Node>& data,
                                      const std::vector<int64_t>& padsBegin,
                                      const std::vector<int64_t>& padsEnd,
                                      float argPadValue,
                                      ov::test::utils::PadMode padMode) const override;
};
}  // namespace test
}  // namespace ov
