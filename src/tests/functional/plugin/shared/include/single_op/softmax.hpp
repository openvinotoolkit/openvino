// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/common_utils.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

namespace aux {

template <class AxisType>
using SoftMaxTestParams = std::tuple<ElementType,   // netPrecision
                                     ElementType,   // inPrecision
                                     ElementType,   // outPrecision
                                     InputShape,    // Dynamic shape + Target static shapes
                                     AxisType,      // axis
                                     TargetDevice,  // targetDevice
                                     Config         // config
                                     >;

template <class AxisType, class SoftmaxOpType>
class SoftMaxLayerTestBase : public testing::WithParamInterface<SoftMaxTestParams<AxisType>>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SoftMaxTestParams<AxisType>>& obj) {
        ElementType netType, inType, outType;
        InputShape shapes;
        AxisType axis;
        TargetDevice targetDevice;
        Config config;
        std::tie(netType, inType, outType, shapes, axis, targetDevice, config) = obj.param;

        std::ostringstream result;
        result << "NetType=" << netType << "_";
        result << "InType=" << inType << "_";
        result << "OutType=" << outType << "_";
        result << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        result << "TS=";
        for (const auto& item : shapes.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "Axis=" << axis << "_";
        result << "Device=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        InputShape shapes;
        ElementType ngPrc;
        AxisType axis;

        std::tie(ngPrc, inType, outType, shapes, axis, targetDevice, configuration) = this->GetParam();
        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, shape));

        const auto softMax = std::make_shared<SoftmaxOpType>(params.at(0), axis);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(softMax)};

        // TODO: This workaround is needed as there is no full support for f16 type in the reference implementation
        if (ngPrc == element::Type_t::f16) {
            abs_threshold = 0.005;
        }

        function = std::make_shared<ov::Model>(results, params, "softMax");
    }
};

}  // namespace aux

using SoftMax1LayerTest = aux::SoftMaxLayerTestBase<size_t, ov::op::v1::Softmax>;
using SoftMax8LayerTest = aux::SoftMaxLayerTestBase<int64_t, ov::op::v8::Softmax>;

using SoftMaxLayerTest = SoftMax1LayerTest;

}  // namespace subgraph
}  // namespace test
}  // namespace ov
