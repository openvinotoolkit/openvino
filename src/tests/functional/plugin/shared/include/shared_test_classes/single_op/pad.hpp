// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<int64_t>,      // padsBegin
        std::vector<int64_t>,      // padsEnd
        float,                     // argPadValue
        ov::op::PadMode,           // padMode
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
    virtual std::shared_ptr<ov::Node> create_pad_op(const std::shared_ptr<ov::Node>&,
                                                    const std::shared_ptr<ov::Node>&,
                                                    const std::shared_ptr<ov::Node>&,
                                                    const std::shared_ptr<ov::Node>&,
                                                    ov::op::PadMode) const;
};

class Pad12LayerTest : public PadLayerTest {
    std::shared_ptr<ov::Node> create_pad_op(const std::shared_ptr<ov::Node>&,
                                            const std::shared_ptr<ov::Node>&,
                                            const std::shared_ptr<ov::Node>&,
                                            const std::shared_ptr<ov::Node>&,
                                            ov::op::PadMode) const override;
};
}  // namespace test
}  // namespace ov
