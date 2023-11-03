// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<std::vector<InputShape>,  // Input shapes
                   std::tuple<bool, int>,    // Is flattened and axis
                   bool,                     // Sorted
                   ElementType               // Data precision
                   >
    UniqueDynamicGPUTestParams;

class UniqueLayerDynamicGPUTest : public testing::WithParamInterface<UniqueDynamicGPUTestParams>,
                                  virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<UniqueDynamicGPUTestParams>& obj) {
        std::vector<InputShape> inputShapes;
        std::tuple<bool, int> flatOrAxis;
        bool sorted;
        ElementType dataPrecision;
        std::tie(inputShapes, flatOrAxis, sorted, dataPrecision) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (size_t i = 0lu; i < inputShapes.size(); i++) {
            result << ov::test::utils::partialShape2str({inputShapes[i].first})
                   << (i < inputShapes.size() - 1lu ? "_" : "");
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < inputShapes.size(); j++) {
                result << ov::test::utils::vec2str(inputShapes[j].second[i])
                       << (j < inputShapes.size() - 1lu ? "_" : "");
            }
            result << "}_";
        }

        if (!std::get<0>(flatOrAxis)) {
            result << "axis=" << std::get<1>(flatOrAxis) << "_";
        } else {
            result << "flattened"
                   << "_";
        }
        result << "sorted=" << (sorted ? "True" : "False") << "_";
        result << "dataPrc=" << dataPrecision;

        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        std::tuple<bool, int> flatOrAxis;
        bool sorted, flattened;
        int axis;
        ElementType dataPrecision;

        std::tie(inputShapes, flatOrAxis, sorted, dataPrecision) = this->GetParam();
        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes(inputShapes);
        flattened = std::get<0>(flatOrAxis);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(dataPrecision, shape));
        }
        params[0]->set_friendly_name("data");
        std::shared_ptr<ov::Node> uniqueNode;
        if (flattened) {
            uniqueNode = std::make_shared<ov::op::v10::Unique>(params[0], sorted);
        } else {
            axis = std::get<1>(flatOrAxis);
            uniqueNode = std::make_shared<ov::op::v10::Unique>(
                params[0],
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape({1}), {axis}),
                sorted);
        }

        // Need to create results for all outputs
        ngraph::ResultVector results;
        for (auto i = 0U; i < uniqueNode->get_output_size(); ++i) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(uniqueNode->output(i)));
        }

        function = std::make_shared<ngraph::Function>(results, params, "Unique");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (funcInput.get_node()->get_friendly_name() == "data") {
                int32_t range = std::accumulate(targetInputStaticShapes[0].begin(),
                                                targetInputStaticShapes[0].end(),
                                                1,
                                                std::multiplies<size_t>());
                tensor = utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                       targetInputStaticShapes[0],
                                                       range,
                                                       -range / 2,
                                                       1);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(UniqueLayerDynamicGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {

const std::vector<ElementType> dataPrecision = {
    ElementType::f16,
    ElementType::i32,
};

std::vector<std::tuple<bool, int>> flatOrAxis{{true, 0}, {false, 0}, {false, 1}, {false, -1}};

std::vector<bool> sorted{true, false};

std::vector<std::vector<InputShape>> getStaticShapes() {
    return {
        {{{}, {{7, 2, 3}}}},
        {{{}, {{7, 2, 3, 5}}}},
        {{{}, {{7, 2, 3, 5, 4}}}},
    };
}

INSTANTIATE_TEST_SUITE_P(smoke_static,
                         UniqueLayerDynamicGPUTest,
                         ::testing::Combine(::testing::ValuesIn(getStaticShapes()),
                                            ::testing::ValuesIn(flatOrAxis),
                                            ::testing::ValuesIn(sorted),
                                            ::testing::ValuesIn(dataPrecision)),
                         UniqueLayerDynamicGPUTest::getTestCaseName);

std::vector<std::vector<InputShape>> getDynamicShapes() {
    return {
        {{{ov::Dimension(2, 15), -1, -1, -1},                                       // Dynamic shape
          {{8, 3, 3, 3}, {6, 5, 2, 5}, {4, 7, 1, 11}, {2, 9, 3, 4}}}},              // Target shapes
        {{{-1, -1, -1, -1, -1},                                                     // Dynamic shape
          {{1, 2, 1, 13, 2}, {3, 4, 7, 2, 2}, {5, 6, 3, 5, 2}, {7, 8, 4, 4, 2}}}},  // Target shapes
    };
}

INSTANTIATE_TEST_SUITE_P(smoke_dynamic,
                         UniqueLayerDynamicGPUTest,
                         ::testing::Combine(::testing::ValuesIn(getDynamicShapes()),
                                            ::testing::ValuesIn(flatOrAxis),
                                            ::testing::ValuesIn(sorted),
                                            ::testing::ValuesIn(dataPrecision)),
                         UniqueLayerDynamicGPUTest::getTestCaseName);

}  // namespace
}  // namespace GPULayerTestsDefinitions
