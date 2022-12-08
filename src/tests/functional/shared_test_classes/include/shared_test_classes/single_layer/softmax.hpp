// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_functions/builders.hpp"

#include "common_test_utils/common_utils.hpp"

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
    static std::string getTestCaseName(const testing::TestParamInfo<SoftMaxTestParams<AxisType>> &obj) {
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
        result << "IS=" << CommonTestUtils::partialShape2str({shapes.first}) << "_";
        result << "TS=";
        for (const auto& item : shapes.second) {
            result << CommonTestUtils::vec2str(item) << "_";
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

        const auto params = ngraph::builder::makeDynamicParams(ngPrc, inputDynamicShapes);
        const auto paramOuts =
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto softMax = std::make_shared<SoftmaxOpType>(paramOuts.at(0), axis);
        const ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(softMax)};

        // TODO: This workaround is needed as there is no full support for f16 type in the reference implementation
        if (ngPrc == element::Type_t::f16) {
            abs_threshold = 0.005;
        }

        function = std::make_shared<ngraph::Function>(results, params, "softMax");
    }
};

} // namespace aux

using SoftMax1LayerTest = aux::SoftMaxLayerTestBase<size_t, ngraph::opset1::Softmax>;
using SoftMax8LayerTest = aux::SoftMaxLayerTestBase<int64_t, ngraph::opset8::Softmax>;

using SoftMaxLayerTest = SoftMax1LayerTest;

}  // namespace subgraph
}  // namespace test
}  // namespace ov
