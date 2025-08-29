// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/scatter_elements_update.hpp"

namespace {
using ScatterUpdateShapes = std::vector<ov::test::InputShape>;
using IndicesValues = std::vector<std::int64_t>;

enum class Scatterupdate_type {
    Basic,
    ND,
    Elements
};

struct ScatterUpdateLayerParams {
    ScatterUpdateShapes inputShapes;
    IndicesValues indicesValues;
    Scatterupdate_type  scType; // scatter update type
};

typedef std::tuple<
    ScatterUpdateLayerParams,
    ov::element::Type,        // input precision
    ov::element::Type         // indices precision
> ScatterUpdateParams;

class ScatterUpdateLayerGPUTest : public testing::WithParamInterface<ScatterUpdateParams>,
                                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ScatterUpdateParams>& obj) {
        const auto& [scatterParams, model_type, idx_type] = obj.param;
        const auto inputShapes = scatterParams.inputShapes;
        const auto indicesValues = scatterParams.indicesValues;
        const auto scType = scatterParams.scType;

        std::ostringstream result;
        result << model_type << "_IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({ shape.first }) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            for (const auto& targetShape : shape.second) {
                result << ov::test::utils::vec2str(targetShape) << "_";
            }
            result << ")_";
        }
        result << "indices_values=" << ov::test::utils::vec2str(indicesValues);
        result << "_idx_precision=" << idx_type;
        result << "_scatter_mode=";
        switch (scType) {
            case Scatterupdate_type::ND:
                result << "ScatterNDUpdate_";
                break;
            case Scatterupdate_type::Elements:
                result << "ScatterElementsUpdate_";
                break;
            case Scatterupdate_type::Basic:
            default:
                result << "ScatterUpdate_";
        }
        result << "trgDev=GPU";
        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            const auto& model_type = funcInput.get_element_type();
            const auto& targetShape = targetInputStaticShapes[i];
            ov::Tensor tensor;
            if (i == 1) {
                tensor = ov::Tensor{ model_type, targetShape };
                const auto indicesVals = std::get<0>(this->GetParam()).indicesValues;
                if (model_type == ov::element::i32) {
                    auto data = tensor.data<std::int32_t>();
                    for (size_t i = 0; i < tensor.get_size(); ++i) {
                        data[i] = static_cast<std::int32_t>(indicesVals[i]);
                    }
                } else if (model_type == ov::element::i64) {
                    auto data = tensor.data<std::int64_t>();
                    for (size_t i = 0; i < tensor.get_size(); ++i) {
                        data[i] = indicesVals[i];
                    }
                } else {
                    OPENVINO_THROW("GatherNDUpdate. Unsupported indices precision: ", model_type);
                }
            } else {
                if (model_type.is_real()) {
                    ov::test::utils::InputGenerateData in_data;
                    in_data.start_from = 0;
                    in_data.range = 10;
                    in_data.resolution = 1000;
                    tensor = ov::test::utils::create_and_fill_tensor(model_type, targetShape, in_data);
                } else {
                    tensor = ov::test::utils::create_and_fill_tensor(model_type, targetShape);
                }
            }
            inputs.insert({ funcInput.get_node_shared_ptr(), tensor });
        }
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [scatterParams, model_type, idx_type] = this->GetParam();
        const auto inputShapes = scatterParams.inputShapes;
        const auto scType = scatterParams.scType;

        init_input_shapes({inputShapes[0], inputShapes[1], inputShapes[2]});


        ov::ParameterVector dataParams{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2])};

        auto indicesParam = std::make_shared<ov::op::v0::Parameter>(idx_type, inputDynamicShapes[1]);
        dataParams[0]->set_friendly_name("Param_1");
        indicesParam->set_friendly_name("Param_2");
        dataParams[1]->set_friendly_name("Param_3");

        std::shared_ptr<ov::Node> scatter;
        switch (scType) {
            case Scatterupdate_type::ND: {
                scatter = std::make_shared<ov::op::v3::ScatterNDUpdate>(dataParams[0], indicesParam, dataParams[1]);
                break;
            }
            case Scatterupdate_type::Elements: {
                auto axis = ov::op::v0::Constant::create(ov::element::i32, inputShapes[3].first.get_shape(), inputShapes[3].second[0]);
                scatter = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dataParams[0], indicesParam, dataParams[1], axis);
                break;
            }
            case Scatterupdate_type::Basic:
            default: {
                auto axis = ov::op::v0::Constant::create(ov::element::i32, inputShapes[3].first.get_shape(), inputShapes[3].second[0]);
                scatter = std::make_shared<ov::op::v3::ScatterUpdate>(dataParams[0], indicesParam, dataParams[1], axis);
            }
        }

        ov::ParameterVector allParams{ dataParams[0], indicesParam, dataParams[1] };

        auto makeFunction = [](ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
            ov::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));

            return std::make_shared<ov::Model>(results, params, "ScatterUpdateLayerGPUTest");
        };
        function = makeFunction(allParams, scatter);
    }
};

TEST_P(ScatterUpdateLayerGPUTest, Inference) {
    run();
}

const std::vector<ScatterUpdateLayerParams> scatterNDParams = {
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{-1, -1, -1, -1, -1}, {{10, 9, 10, 9, 10}, {10, 1, 11, 2, 5}, {10, 15, 8, 1, 7}}},
            {{2, 2, 1}, {{2, 2, 1}, {2, 2, 1}, {2, 2, 1}}},
            {{-1, -1, -1, -1, -1, -1}, {{2, 2, 9, 10, 9, 10}, {2, 2, 1, 11, 2, 5}, {2, 2, 15, 8, 1, 7}}},
        },
        IndicesValues{ 5, 6, 2, 8 },
        Scatterupdate_type::ND
    },
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{-1, -1, -1, -1}, {{ 10, 9, 9, 11 }, { 7, 5, 3, 12 }, { 3, 4, 9, 8 }}},
            {{2, 3}, {{2, 3}, {2, 3}, {2, 3}}},
            {{-1, -1}, {{2, 11}, {2, 12}, {2, 8}}}
        },
        IndicesValues{ 0, 1, 1, 2, 2, 2 },
        Scatterupdate_type::ND
    },
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{{3, 10}, -1, {3, 9}, -1}, {{ 10, 9, 9, 11 }, { 7, 5, 3, 12 }, { 3, 4, 9, 8 }}},
            {{2, 3}, {{2, 3}, {2, 3}, {2, 3}}},
            {{{2, 4}, -1}, {{2, 11}, {2, 12}, {2, 8}}}
        },
        IndicesValues{ 0, 1, 1, 2, 2, 2 },
        Scatterupdate_type::ND
    },
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{{3, 10}, {4, 11}, {3, 9}, {8, 15}}, {{ 10, 9, 9, 11 }, { 7, 5, 3, 12 }, { 3, 4, 9, 8 }}},
            {{2, 3}, {{2, 3}, {2, 3}, {2, 3}}},
            {{{2, 4}, -1}, {{2, 11}, {2, 12}, {2, 8}}}
        },
        IndicesValues{ 0, 1, 1, 2, 2, 2 },
        Scatterupdate_type::ND
    },
};

const std::vector<ScatterUpdateLayerParams> scatterElementsParams = {
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{-1, -1, -1, -1, -1}, {{10, 9, 10, 9, 10}, {10, 5, 11, 4, 5}, {10, 15, 8, 1, 7}}},
            {{-1, -1, -1, -1, -1 }, {{3, 2, 1, 2, 1}, {3, 2, 1, 2, 1}, {3, 2, 1, 2, 1}}},
            {{-1, -1, -1, -1, -1 }, {{3, 2, 1, 2, 1}, {3, 2, 1, 2, 1}, {3, 2, 1, 2, 1}}},
            {{1}, {{1}}}
        },
        IndicesValues{ 5, 6, 2, 8, 5, 6, 2, 8, 5, 6, 2, 8 },
        Scatterupdate_type::Elements
    },
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{-1, -1, -1, -1}, {{ 10, 9, 9, 11 }, { 7, 5, 3, 12 }, { 3, 4, 9, 8 }}},
            {{-1, -1, -1, -1}, {{3, 1, 2, 3}, {3, 1, 2, 3}, {3, 1, 2, 3}}},
            {{-1, -1, -1, -1}, {{3, 1, 2, 3}, {3, 1, 2, 3}, {3, 1, 2, 3}}},
            {{1}, {{1}}}
        },
        IndicesValues{ 0, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2 },
        Scatterupdate_type::Elements
    },
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{{3, 10}, -1, {3, 9}, -1}, {{ 10, 9, 9, 11 }, { 7, 5, 3, 12 }, { 3, 4, 9, 8 }}},
            {{2, -1, 3, -1}, {{2, 1, 3, 1}, {2, 1, 3, 1}, {2, 1, 3, 1}}},
            {{2, -1, 3, -1}, {{2, 1, 3, 1}, {2, 1, 3, 1}, {2, 1, 3, 1}}},
            {{1}, {{1}}}
        },
        IndicesValues{ 0, 1, 1, 2, 2, 2 },
        Scatterupdate_type::Elements
    },
};

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
};

const std::vector<ov::element::Type> constantPrecisions = {
    ov::element::i32,
};

const std::vector<ScatterUpdateLayerParams> scatterUpdate_EmptyInput1_2Params = {
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{-1, -1, -1, -1}, {{ 100, 256, 14, 14 }}},
            {{-1}, {{ 0 }}},
            {{-1, 256, 14, 14}, {{ 0, 256, 14, 14 }}},
            {{1}, {{0}}}
        },
        IndicesValues{ 0 },
        Scatterupdate_type::Basic
    },
};

const std::vector<ScatterUpdateLayerParams> scatterNDUpdate_EmptyInput1_2Params = {
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{-1, -1, -1, -1}, {{ 100, 256, 14, 14 }}},
            {{-1, 1}, {{ 0, 1 }}},
            {{-1, 256, 14, 14}, {{ 0, 256, 14, 14 }}}
        },
        IndicesValues{ 0 },
        Scatterupdate_type::ND
    },
};

const std::vector<ScatterUpdateLayerParams> scatterElementsUpdate_EmptyInput1_2Params = {
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{-1, -1, -1, -1}, {{ 100, 256, 14, 14 }}},
            {{-1, -1, 14, 14}, {{ 0, 256, 14, 14 }}},
            {{-1, 256, 14, 14}, {{ 0, 256, 14, 14 }}},
            {{1}, {{0}}}
        },
        IndicesValues{ 0 },
        Scatterupdate_type::Elements
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_ScatterNDUpdate_CompareWithRefs_dynamic, ScatterUpdateLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(scatterNDParams),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(constantPrecisions)),
    ScatterUpdateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterElementsUpdate_CompareWithRefs_dynamic, ScatterUpdateLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(scatterElementsParams),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(constantPrecisions)),
    ScatterUpdateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterUpdate_EmptyInput1_2_CompareWithRefs_dynamic, ScatterUpdateLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(scatterUpdate_EmptyInput1_2Params),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(constantPrecisions)),
    ScatterUpdateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterNDUpdate_EmptyInput1_2_CompareWithRefs_dynamic, ScatterUpdateLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(scatterNDUpdate_EmptyInput1_2Params),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(constantPrecisions)),
    ScatterUpdateLayerGPUTest::getTestCaseName);

// ScatterELementsUpdate doesn't support dynamic shape yet. Need to enable when it supports.
INSTANTIATE_TEST_SUITE_P(smoke_ScatterElementsUpdate_EmptyInput1_2_CompareWithRefs_dynamic, ScatterUpdateLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(scatterElementsUpdate_EmptyInput1_2Params),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(constantPrecisions)),
    ScatterUpdateLayerGPUTest::getTestCaseName);
} // namespace
