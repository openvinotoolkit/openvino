// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/gather.hpp"

namespace {
using ov::test::InputShape;

struct GatherShapeParams {
    InputShape inputShapes;
    InputShape targetShapes;
    int axis;
    int batch_dims;
};

typedef std::tuple<
        GatherShapeParams,
        ov::element::Type,                     // Network precision
        bool,                            // Is const Indices
        bool                             // Is const Axis
> GatherGPUTestParams;

class GatherGPUTest : public testing::WithParamInterface<GatherGPUTestParams>,
                      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherGPUTestParams> obj) {
        const auto& [Shapes, model_type, isIndicesConstant, isAxisConstant] = obj.param;

        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::partialShape2str({Shapes.inputShapes.first}) << "_";
        for (size_t i = 0lu; i < Shapes.inputShapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(Shapes.inputShapes.second[i]) << "_";
            result << "}_";
        }
        result << "TS=(";
        result << ov::test::utils::partialShape2str({Shapes.targetShapes.first}) << "_";
        for (size_t i = 0lu; i < Shapes.targetShapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(Shapes.targetShapes.second[i]) << "_";
            result << "}_";
        }
        result << "axis=" << Shapes.axis << "_";
        result << "batchDims=" << Shapes.batch_dims << "_";
        result << "netPrc=" << model_type << "_";
        result << "constIdx=" << (isIndicesConstant ? "True" : "False") << "_";
        result << "constAx=" << (isAxisConstant ? "True" : "False") << "_";

        return result.str();
    }

protected:
    void SetUp() override {
        const auto int_model_type = ov::element::i32;

        const auto& [Shapes, model_type, isIndicesConstant, isAxisConstant] = this->GetParam();
        const int axis = Shapes.axis;
        const int batchDims = Shapes.batch_dims;
        targetDevice = ov::test::utils::DEVICE_GPU;
        std::shared_ptr<ov::Node> indicesNode;
        std::shared_ptr<ov::Node> gatherNode;
        std::shared_ptr<ov::Node> axisNode;

        if (isIndicesConstant) {
            init_input_shapes({Shapes.inputShapes});
        } else { // Not being tested because currently parameter targetshape is not supported
            init_input_shapes({Shapes.inputShapes, Shapes.targetShapes});
        }

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};
        params.back()->set_friendly_name("data");

        if (isIndicesConstant) {
            auto dimsize = Shapes.inputShapes.second[0].size();
            int64_t idx_range = INT64_MAX;
            auto axis_norm = axis < 0 ? axis + dimsize : axis;
            for (size_t i = 0; i < Shapes.inputShapes.second.size(); ++i) {
                idx_range = std::min(static_cast<int64_t>(Shapes.inputShapes.second[i][axis_norm]), idx_range);
            }

            auto indices_tensor = ov::test::utils::create_and_fill_tensor(ov::element::i64, Shapes.targetShapes.second[0], idx_range - 1, 0);
            indicesNode = std::make_shared<ov::op::v0::Constant>(indices_tensor);
        } else {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(int_model_type, inputDynamicShapes[1]));
            params.back()->set_friendly_name("indices");
        }

        if (isAxisConstant) {
            axisNode = std::make_shared<ov::op::v0::Constant>(int_model_type, ov::Shape({1}), std::vector<int64_t>{axis});
        } else {
            inputDynamicShapes.push_back({1});
            for (size_t i = 0lu; i < targetStaticShapes.size(); i++) {
                targetStaticShapes[i].push_back({1});
            }
            params.push_back(std::make_shared<ov::op::v0::Parameter>(int_model_type, inputDynamicShapes[2]));
            params.back()->set_friendly_name("axis");
        }

        gatherNode = std::make_shared<ov::op::v7::Gather>(params[0],
                                                          isIndicesConstant ? indicesNode : params[1],
                                                          isAxisConstant    ? axisNode
                                                                            : isIndicesConstant ? params[1]
                                                                            : params[2],
                                                          batchDims);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gatherNode)};
        function = std::make_shared<ov::Model>(results, params, "Gather");
    }
};

TEST_P(GatherGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::i32,
    ov::element::i64,
    ov::element::i8
};

const std::vector<GatherShapeParams> dynamicInputShapeConstTargetShape = {
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1}), {{2, 3, 4}, {3, 4, 5}}),
        ov::test::InputShape(ov::PartialShape({}), {{2}}),
        1, 0
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1}), {{1, 2, 3, 4}, {1, 3, 4, 5}}),
        ov::test::InputShape(ov::PartialShape({}), {{}}),
        2, 0
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1}), {{2, 7, 8, 9}, {2, 7, 4, 8}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 1}}),
        2, 1
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1}), {{2, 1, 3, 3}, {2, 1, 10, 11}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 1}}),
        3, 2
    },
    {
        ov::test::InputShape(ov::PartialShape({8, -1, -1, 2}), {{8, 2, 3, 2}, {8, 4, 5, 2}}),
        ov::test::InputShape(ov::PartialShape({}), {{}}),
        0, 0
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{2, 6, 7, 8, 9}, {2, 6, 9, 1, 2}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 6}}),
        3, 1
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{2, 4, 2, 2, 3}, {2, 4, 8, 9, 10}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 4}}),
        2, 1
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{3, 4, 2, 2, 3}, {3, 4, 8, 9, 10}}),
        ov::test::InputShape(ov::PartialShape({}), {{3, 4, 3}}),
        3, 2
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{2, 4, 2, 2, 3}, {2, 4, 8, 9, 10}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 4}}),
        2, 2
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1, -1}), {{2, 4, 2, 3, 1, 3}, {2, 4, 7, 8, 9, 10}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 4}}),
        2, 2
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_input_shapes_const_target_shapes, GatherGPUTest,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapeConstTargetShape),    // input shapes
                    ::testing::ValuesIn(model_types),                          // network precision
                    ::testing::Values(true),                                   // is const indices
                    ::testing::Values(true)),                                  // is const axis
                GatherGPUTest::getTestCaseName);
} // namespace
