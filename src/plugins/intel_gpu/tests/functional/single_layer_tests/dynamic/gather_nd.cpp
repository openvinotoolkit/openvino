// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/gather_nd.hpp"

namespace {
using ov::test::InputShape;

struct GatherNDShapeParams {
    InputShape inputShapes;
    InputShape targetShapes;
    int batch_dims;
};

typedef std::tuple<
        GatherNDShapeParams,
        ov::element::Type,     // Model type
        bool                   // Is const Indices
> GatherNDGPUTestParams;


class GatherNDGPUTest : public testing::WithParamInterface<GatherNDGPUTestParams>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherNDGPUTestParams> obj) {
        const auto& [Shapes, model_type, isIndicesConstant] = obj.param;

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
        result << "batchDims=" << Shapes.batch_dims << "_";
        result << "netPrc=" << model_type << "_";
        result << "constIdx=" << (isIndicesConstant ? "True" : "False") << "_";

        return result.str();
    }

protected:
    void SetUp() override {
        const auto intInputsPrecision = ov::element::i32;

        const auto& [Shapes, model_type, isIndicesConstant] = this->GetParam();
        const int batchDims = Shapes.batch_dims;
        targetDevice = ov::test::utils::DEVICE_GPU;
        std::shared_ptr<ov::Node> indicesNode;
        std::shared_ptr<ov::Node> gather_ndNode;

        if (isIndicesConstant) {
            init_input_shapes({Shapes.inputShapes});
        } else { // Not being tested because currently parameter targetshape is not supported
            init_input_shapes({Shapes.inputShapes, Shapes.targetShapes});
        }

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};
        params.back()->set_friendly_name("data");

        if (isIndicesConstant) {
            int64_t idx_range = INT64_MAX;
            for (size_t i = 0; i < Shapes.inputShapes.second.size(); ++i) {
                for (size_t j = 0; j < Shapes.inputShapes.second[i].size(); ++j) {
                    idx_range = std::min(static_cast<int64_t>(Shapes.inputShapes.second[i][j]), idx_range);
                }
            }
            auto indices_tensor = ov::test::utils::create_and_fill_tensor(ov::element::i64, Shapes.targetShapes.second[0], idx_range - 1, 0);
            indicesNode = std::make_shared<ov::op::v0::Constant>(indices_tensor);
        } else {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(intInputsPrecision, inputDynamicShapes[1]));
            params.back()->set_friendly_name("indices");
        }

        gather_ndNode = std::make_shared<ov::op::v8::GatherND>(params[0],
                                                          isIndicesConstant ? indicesNode : params[1],
                                                          batchDims);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather_ndNode)};
        function = std::make_shared<ov::Model>(results, params, "GatherND");
    }
};

TEST_P(GatherNDGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i32
};

const std::vector<GatherNDShapeParams> dynamicInputShapeConstTargetShape = {
    {
        ov::test::InputShape(ov::PartialShape({-1, -1}), {{2, 2}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 1}}),
        1
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1}), {{10, 14}}),
        ov::test::InputShape(ov::PartialShape({}), {{3, 2}}),
        0
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1}), {{2, 3, 4}, {3, 4, 5}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 1}}),
        0
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1}), {{2, 7, 8, 9}, {2, 7, 4, 8}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 1}}),
        1
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{15, 12, 20, 15, 2}, {15, 12, 18, 7, 17}}),
        ov::test::InputShape(ov::PartialShape({}), {{15, 12, 5, 9, 1, 3}}),
        2
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1, -1}), {{4, 3, 2, 5, 5, 2}, {4, 3, 2, 5, 7, 2}}),
        ov::test::InputShape(ov::PartialShape({}), {{4, 3, 2, 5, 6, 2}}),
        4
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_input_shapes_const_target_shapes, GatherNDGPUTest,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapeConstTargetShape),    // input shapes
                    ::testing::ValuesIn(model_types),          // network precision
                    ::testing::Values(true)),                     // is const indices
                GatherNDGPUTest::getTestCaseName);
} // namespace
