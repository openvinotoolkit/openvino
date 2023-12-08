// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "ov_models/builders.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

using GroupNormalizationTestParams = std::tuple<ElementType,   // netPrecision
                                                ElementType,   // inPrecision
                                                ElementType,   // outPrecision
                                                InputShape,    // Dynamic shape + Target static shapes
                                                std::int64_t,  // num_groups
                                                double,        // epsilon
                                                TargetDevice,  // targetDevice
                                                Config         // config
                                                >;

class GroupNormalizationTest : public testing::WithParamInterface<GroupNormalizationTestParams>,
                               virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupNormalizationTestParams> &obj) {
        ElementType netType, inType, outType;
        InputShape shapes;
        std::int64_t num_groups;
        double epsilon;
        TargetDevice targetDevice;
        Config config;
        std::tie(netType, inType, outType, shapes, num_groups, epsilon, targetDevice, config) = obj.param;

        std::ostringstream result;
        result << "NetType=" << netType << "_";
        result << "InType=" << inType << "_";
        result << "OutType=" << outType << "_";
        result << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        result << "TS=";
        for (const auto& item : shapes.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "NumGroups=" << num_groups << "_";
        result << "Epsilon=" << epsilon << "_";
        result << "Device=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        InputShape shapes;
        ElementType ngPrc;
        std::int64_t num_groups;
        double epsilon;

        std::tie(ngPrc, inType, outType, shapes, num_groups, epsilon, targetDevice, configuration) = this->GetParam();
        InputShape biasInputShape = ExtractBiasShape(shapes);
        init_input_shapes({shapes, biasInputShape, biasInputShape});
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, shape));

        const auto groupNormalization = std::make_shared<ov::op::v12::GroupNormalization>(
            params.at(0),
            params.at(1),
            params.at(2),
            num_groups,
            epsilon);
        const ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(groupNormalization)};

        // TODO: This workaround is needed as there is no full support for f16 type in the reference implementation
        if (ngPrc == element::Type_t::f16) {
            abs_threshold = 0.007;
        }

        function = std::make_shared<ngraph::Function>(results, params, "GroupNormalization");
    }

    InputShape ExtractBiasShape(const InputShape& shape) {
        std::vector<ov::Shape> biasShape;
        std::transform(shape.second.cbegin(), shape.second.cend(), std::back_inserter(biasShape),
                       [](const ov::Shape& s)->ov::Shape { return {s[1]}; });
        InputShape biasInputShape {
            shape.first.is_dynamic() ? ov::PartialShape{shape.first[1]} : shape.first,
            std::move(biasShape)
        };
        return biasInputShape;
    }
};

}  // namespace subgraph
}  // namespace test
}  // namespace ov
