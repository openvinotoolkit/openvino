// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"


#include "openvino/opsets/opset13.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "openvino/pass/manager.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/matmul.hpp"

#include "intel_gpu/runtime/execution_config.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<ov::element::Type,                // netPrecision
                   std::vector<InputShape>,          // shape
                   bool,                             // is_causal
                   bool,                             // has_attn
                   bool,                             // has_scale
                   std::vector<std::vector<int64_t>> // input_transpose
                   > ScaledAttnGPUTestParams;

class ScaledAttnLayerGPUTest : public testing::WithParamInterface<ScaledAttnGPUTestParams>,
                               virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ScaledAttnGPUTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void transpose_prepare(std::vector<InputShape>& shapes, const std::vector<std::vector<int64_t>>& input_transpose);
    bool is_causal;
    bool has_attn;
    bool has_scale;
};

std::string ScaledAttnLayerGPUTest::getTestCaseName(const testing::TestParamInfo<ScaledAttnGPUTestParams>& obj) {
    ov::element::Type inType;
    std::vector<InputShape> inputShapes;
    std::vector<std::vector<int64_t>> input_transpose;
    bool is_causal;
    bool has_attn;
    bool has_scale;
    bool transpose_enable;
    std::tie(inType, inputShapes, is_causal, has_attn, has_scale, input_transpose) = obj.param;

    transpose_enable = (input_transpose.size() != 0);
    std::ostringstream result;
    result << "netPRC=" << inType << "_";
    result << "IS=";
    for (const auto& inputShape : inputShapes) {
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
    }
    result << "TS=";
    for (const auto& shapes : inputShapes) {
        for (const auto& shape : shapes.second) {
            result << ov::test::utils::vec2str(shape);
            result << "_";
        }
    }
    result << "is_causal=" << is_causal << "_";
    result << "has_attn=" << has_attn << "_";
    result << "has_scale=" << has_scale << "_";
    result << "with_transpose" << transpose_enable << "_";

    return result.str();
}

void ScaledAttnLayerGPUTest::SetUp() {
    ov::element::Type inType;
    std::vector<InputShape> inputShapes;
    std::vector<std::vector<int64_t>> input_transpose;

    targetDevice = ov::test::utils::DEVICE_GPU;

    std::tie(inType, inputShapes, is_causal, has_attn, has_scale, input_transpose) = this->GetParam();

    transpose_prepare(inputShapes, input_transpose);
    init_input_shapes(inputShapes);
    ov::ParameterVector inputParams;
    // q, k, v
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[2]));
    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");
    // special case: only scale but no attn
    if (!has_attn && has_scale) {
        // attention_mask：[1]
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{}));
        inputParams.back()->set_friendly_name("attention_mask");
        // scale：[1]
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{1}));
        inputParams.back()->set_friendly_name("scale");
    } else {
        if (has_attn) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[3]));
            inputParams.back()->set_friendly_name("attention_mask");
        }
        if (has_scale) {
            // scale：[1]
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{1}));
            inputParams.back()->set_friendly_name("scale");
        }
    }

    ov::OutputVector inputParams_transpose;
    for (size_t i = 0; i < inputParams.size(); i++) {
        inputParams_transpose.push_back(inputParams[i]);
    }
    if (input_transpose.size() != 0) {
        // deal with transpose.
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, input_transpose[0]);
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(inputParams[0], tranpose_a_const);
        tranpose_a->set_friendly_name("tranpose_a");
        inputParams_transpose[0] = tranpose_a;

        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, input_transpose[1]);
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(inputParams[1], tranpose_b_const);
        tranpose_b->set_friendly_name("tranpose_b");
        inputParams_transpose[1] = tranpose_b;

        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, input_transpose[2]);
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(inputParams[2], tranpose_c_const);
        tranpose_c->set_friendly_name("tranpose_c");
        inputParams_transpose[2] = tranpose_c;
    }

    ov::OutputVector inputs;
    for (size_t i = 0; i < inputParams_transpose.size(); i++) {
        inputs.push_back(inputParams_transpose[i]);
    }

    auto sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(inputs, is_causal);
    sdp->set_friendly_name("sdpa");

    auto output = std::make_shared<ov::op::v0::Result>(sdp->output(0));

    function = std::make_shared<ov::Model>(ov::OutputVector{output}, inputParams, "sdpa_model");

    functionRefs = function->clone();
    ov::pass::Manager manager;

    // Decompose ScaledDotProductAttention
    manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    manager.run_passes(functionRefs);

    auto it = std::find_if(inputShapes[1].second.begin(), inputShapes[1].second.end(), [&](const ov::Shape& shape){
        return shape[2] >= 384 || shape[3] >= 128;
    });

    bool has_long_seq = it != inputShapes[1].second.end();
    if (inType == ov::element::f16) {
        if (has_long_seq) {
            abs_threshold = 0.025;
            rel_threshold = 0.025;
        } else {
            abs_threshold = 0.005;
            rel_threshold = 0.005;
        }
    }
}

void ScaledAttnLayerGPUTest::transpose_prepare(std::vector<InputShape>& shapes,
    const std::vector<std::vector<int64_t>>& input_transpose) {
    auto transpose_pshape = [](InputShape& pshapes, const std::vector<int64_t>& order) {
        auto transposed_pshape = ov::PartialShape::dynamic(pshapes.first.rank());
        std::vector<ov::Shape> transposed_cshapes(pshapes.second);
        auto& pshape = pshapes.first;
        auto& cshape = pshapes.second;
        for (size_t i = 0; i < order.size(); i++) {
            transposed_pshape[i] = pshape[order[i]];
            for (size_t j = 0; j < cshape.size(); j++) {
                transposed_cshapes[j][i] = cshape[j][order[i]];
            }
        }

        for (size_t i = 0; i < order.size(); i++) {
            pshape[i] = transposed_pshape[i];
            for (size_t j = 0; j < cshape.size(); j++) {
                cshape[j][i] = transposed_cshapes[j][i];
            }
        }
    };

    if (shapes.empty()) {
        return;
    }

    shapes.insert(shapes.begin()+1, shapes[1]);
    if (input_transpose.empty()) {
        return;
    }

    for (size_t i = 0; i < input_transpose.size(); i++) {
        transpose_pshape(shapes[i], input_transpose[i]);
    }
}

void ScaledAttnLayerGPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    std::vector<ov::Shape> shapes(3);
    shapes[0] = targetInputStaticShapes[0];
    shapes[1] = targetInputStaticShapes[1];
    shapes[2] = targetInputStaticShapes[2];
    if (!has_attn && has_scale) {
        shapes.push_back(ov::Shape{});
        shapes.push_back(ov::Shape{1});
    } else {
        if (has_attn) {
            shapes.push_back(targetInputStaticShapes[3]);
        }
        if (has_scale) {
            shapes.push_back(ov::Shape{1});
        }
    }
    SubgraphBaseTest::generate_inputs(shapes);
}

TEST_P(ScaledAttnLayerGPUTest, CompareWithRefs) {
    ov::element::Type inType;
    std::vector<InputShape> inputShapes;
    std::vector<std::vector<int64_t>> input_transpose;
    bool is_causal;
    bool has_attn;
    bool has_scale;
    std::tie(inType, inputShapes, is_causal, has_attn, has_scale, input_transpose) = this->GetParam();
    run();
}

const std::vector<std::vector<InputShape>> shapes{
    // normal case, shapes of q,k,v are same
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 128},
            {ov::Shape{1, 8, 100, 128}, ov::Shape{1, 8, 1, 128}, ov::Shape{2, 8, 10, 128}}}
        },
        // kv shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 128},
            {ov::Shape{1, 8, 100, 128}, ov::Shape{1, 8, 1, 128}, ov::Shape{2, 8, 10, 128}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, -1},
            {ov::Shape{1, 1, 100, 100}, ov::Shape{1, 1, 1, 1}, ov::Shape{2, 1, 10, 10}}}
        },
    },
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 100, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // kv shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 100, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, -1},
            {ov::Shape{1, 1, 100, 100}, ov::Shape{1, 1, 1, 1}, ov::Shape{2, 1, 10, 10}}}
        },
    },
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 5, -1, 128},
            {ov::Shape{2, 5, 100, 128}, ov::Shape{2, 5, 1, 128}, ov::Shape{2, 5, 387, 128}}}
        },
        // kv shape
        {ov::test::InputShape{ov::PartialShape{-1, 5, -1, 128},
            {ov::Shape{2, 5, 100, 128}, ov::Shape{2, 5, 1, 128}, ov::Shape{2, 5, 387, 128}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, -1},
            {ov::Shape{1, 1, 100, 100}, ov::Shape{1, 1, 1, 1}, ov::Shape{2, 1, 387, 387}}}
        },
    },
    // heads number of kv is 1, attn mask: [B, H, L1, L0+L1]
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 128},
            {ov::Shape{1, 8, 100, 128}, ov::Shape{1, 8, 1, 128}, ov::Shape{2, 8, 10, 128}}}
        },
        // kv shape
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, 128},
            {ov::Shape{1, 1, 100, 128}, ov::Shape{1, 1, 1, 128}, ov::Shape{2, 1, 10, 128}}}
        },
        // attn shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, -1},
            {ov::Shape{1, 8, 100, 100}, ov::Shape{1, 8, 1, 1}, ov::Shape{2, 8, 10, 10}}}
        },
    },
    // large head size
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 256},
            {ov::Shape{1, 8, 7, 256}, ov::Shape{1, 8, 1, 256}, ov::Shape{2, 8, 10, 256}}}
        },
        // kv shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 256},
            {ov::Shape{1, 8, 7, 256}, ov::Shape{1, 8, 1, 256}, ov::Shape{2, 8, 10, 256}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, -1},
            {ov::Shape{1, 1, 7, 7}, ov::Shape{1, 1, 1, 1}, ov::Shape{2, 1, 10, 10}}}
        },
    }
};

const std::vector<std::vector<int64_t>> disable_transpose{};
const std::vector<std::vector<int64_t>> transpose_value{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 2, 1, 3}};
const std::vector<std::vector<int64_t>> transpose_all{{0, 2, 1, 3}, {0, 2, 1, 3}, {0, 2, 1, 3}};

const auto dynamic_shape_params = testing::Combine(testing::Values(ov::element::f16 /*, ov::element::f32 */),
                                                   testing::ValuesIn(shapes),
                                                   testing::Values(true, false),
                                                   testing::Values(true, false),
                                                   testing::Values(true, false),
                                                   testing::ValuesIn({disable_transpose, transpose_value}));

INSTANTIATE_TEST_SUITE_P(smoke_ScaledAttn_GPU,
                         ScaledAttnLayerGPUTest,
                         dynamic_shape_params,
                         ScaledAttnLayerGPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> static_shapes{
    // static shapes
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 100, 128},
            {ov::Shape{1, 8, 100, 128}}}
        },
        // kv shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 100, 128},
            {ov::Shape{1, 8, 100, 128}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{1, 1, 100, 100},
            {ov::Shape{1, 1, 100, 100}}}
        },
    },
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 64, 128},
            {ov::Shape{1, 8, 64, 128}}}
        },
        // kv shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 13, 128},
            {ov::Shape{1, 8, 13, 128}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{1, 1, 64, 13},
            {ov::Shape{1, 1, 64, 13}}}
        },
    },
};

const auto static_shape_params = testing::Combine(testing::Values(ov::element::f16),
                                                  testing::ValuesIn(static_shapes),
                                                  testing::Values(true, false),
                                                  testing::Values(true, false),
                                                  testing::Values(true, false),
                                                  testing::ValuesIn({disable_transpose, transpose_all}));

INSTANTIATE_TEST_SUITE_P(smoke_ScaledAttnStatic_GPU,
                         ScaledAttnLayerGPUTest,
                         static_shape_params,
                         ScaledAttnLayerGPUTest::getTestCaseName);

} // namespace
