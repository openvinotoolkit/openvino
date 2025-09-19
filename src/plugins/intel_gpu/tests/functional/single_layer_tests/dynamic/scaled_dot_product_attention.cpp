// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/opsets/opset13_decl.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "openvino/pass/manager.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/matmul.hpp"

#include "intel_gpu/runtime/execution_config.hpp"
#include "openvino/op/transpose.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<ov::element::Type,                // netPrecision
                   std::vector<InputShape>,          // shape
                   bool,                             // is_causal
                   bool,                             // has_attn
                   bool,                             // is_attn_const
                   bool,                             // has_scale
                   bool,                             // is_scale_const
                   std::vector<std::vector<int64_t>>, // input_transpose
                   bool                             // has_sink
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
    bool is_attn_const;
    bool has_scale;
    bool is_scale_const;
    bool has_sink;
};

std::string ScaledAttnLayerGPUTest::getTestCaseName(const testing::TestParamInfo<ScaledAttnGPUTestParams>& obj) {
    bool transpose_enable;
    const auto& [inType, inputShapes, is_causal, has_attn, is_attn_const, has_scale, is_scale_const, input_transpose, has_sink] = obj.param;

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
    result << "is_attn_const=" << is_attn_const << "_";
    result << "has_scale=" << has_scale << "_";
    result << "is_scale_const=" << is_scale_const << "_";
    result << "with_transpose" << transpose_enable << "_";
    result << "has_sink=" << has_sink << "_";

    return result.str();
}

void ScaledAttnLayerGPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_GPU;

    const auto& [inType, _inputShapes, _is_causal, _has_attn, _is_attn_const, _has_scale, _is_scale_const, input_transpose, _has_sink] = this->GetParam();
    is_causal = _is_causal;
    has_attn = _has_attn;
    is_attn_const = _is_attn_const;
    has_scale = _has_scale;
    is_scale_const = _is_scale_const;
    auto inputShapes = _inputShapes;
    has_sink = _has_sink;

    transpose_prepare(inputShapes, input_transpose);
    init_input_shapes(inputShapes);
    // Q[q_n_heads, seq, head_size]
    // K[k_n_heads, seq, head_size]
    // QK[q_n_heads, seq, seq]
    // Sink[q_n_heads, 1, 1]
    // Sink_broadcast[n_heads, seq, 1];
    // QK_Sink = Concat(QK[q_n_heads, seq, seq], Sink_broadcast[n_heads, seq, 1], axis = -1) => [q_n_heads, seq, seq + 1]
    // Softmax(QK_Sink) : [q_n_heads, seq, seq + 1]
    // Slice(Softmax) => [q_n_heads, seq, seq)
    // ATTN_O = MatMul(Slice, V)

    ov::ParameterVector inputParams;
    // q, k, v
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[2]));
    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");
    if (!has_attn && has_scale) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{}));
        inputParams.back()->set_friendly_name("attention_mask");
        if (!is_scale_const) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{1}));
            inputParams.back()->set_friendly_name("scale");
        }
    } else {
        if (has_attn && !is_attn_const) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[3]));
            inputParams.back()->set_friendly_name("attention_mask");
            if (has_scale && !is_scale_const) {
                inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{1}));
                inputParams.back()->set_friendly_name("scale");
            }
        }
    }


    ov::OutputVector inputParams_transpose;
    for (size_t i = 0; i < inputParams.size(); i++) {
        inputParams_transpose.push_back(inputParams[i]);
    }
    if (has_attn && is_attn_const) {
        auto attn_const = std::make_shared<ov::op::v0::Constant>(inType, ov::Shape{}, 0.0f);
        attn_const->set_friendly_name("attention_mask");
        inputParams_transpose.push_back(attn_const);
        if (has_scale && !is_scale_const) {
            auto scale_param = std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{1});
            scale_param->set_friendly_name("scale");
            inputParams.push_back(scale_param);
            inputParams_transpose.push_back(scale_param);
        }
    }
    if (has_scale && is_scale_const) {
        auto scale_const = std::make_shared<ov::op::v0::Constant>(inType, ov::Shape({1}), 0.35f);
        scale_const->set_friendly_name("scale");
        inputParams_transpose.push_back(scale_const);
    }

    if (input_transpose.size() != 0) {
        auto rank = input_transpose[0].size();
        // deal with transpose.
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{rank}, input_transpose[0]);
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(inputParams[0], tranpose_a_const);
        tranpose_a->set_friendly_name("tranpose_a");
        inputParams_transpose[0] = tranpose_a;

        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{rank}, input_transpose[1]);
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(inputParams[1], tranpose_b_const);
        tranpose_b->set_friendly_name("tranpose_b");
        inputParams_transpose[1] = tranpose_b;

        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{rank}, input_transpose[2]);
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(inputParams[2], tranpose_c_const);
        tranpose_c->set_friendly_name("tranpose_c");
        inputParams_transpose[2] = tranpose_c;
    }

    ov::OutputVector inputs;
    for (size_t i = 0; i < inputParams_transpose.size(); i++) {
        inputs.push_back(inputParams_transpose[i]);
    }
    if (has_sink) {
        size_t num_heads = inputDynamicShapes[0][1].get_length();
        auto sink_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f16, ov::Shape{1, num_heads, 1, 1}, 10.f, 100.f, 1);
        auto sink_const = std::make_shared<ov::op::v0::Constant>(sink_tensor);
        sink_const->set_friendly_name("sink");
        inputs.push_back(sink_const);
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
    ov::pass::Serialize("./ref.xml", "").run_on_model(functionRefs);

    auto it = std::find_if(inputShapes[1].second.begin(), inputShapes[1].second.end(), [&](const ov::Shape& shape){
        return shape[0] >= 128 || shape[2] >= 384 || shape[3] >= 128;
    });

    bool has_diff_head_size = inputShapes[1].first.begin()[3] != inputShapes[2].first.begin()[3];

    bool has_long_seq = it != inputShapes[1].second.end();

    if (inType == ov::element::f16) {
        if (has_diff_head_size && !has_scale) {
            abs_threshold = 0.1;
            rel_threshold = 0.1;
        } else if (has_long_seq) {
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

    if (input_transpose.empty()) {
        return;
    }

    for (size_t i = 0; i < input_transpose.size(); i++) {
        transpose_pshape(shapes[i], input_transpose[i]);
    }
}

void ScaledAttnLayerGPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const auto& model_inputs = function->inputs();
    inputs.clear();
    std::vector<ov::Shape> shapes(3);
    {
        // Generate QKV
//        for (int i = 0; i < 3; ++i) {
//            shapes[i] = targetInputStaticShapes[i];
//            ov::test::utils::InputGenerateData data(0, 8, 32);
//            ov::Tensor data_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f16, shapes[i], data);
//            inputs.insert({model_inputs[i].get_node_shared_ptr(), data_tensor});
//        }
        // Q
        shapes[0] = targetInputStaticShapes[0];
        ov::test::utils::InputGenerateData data0(1.1, 4, 1);
        ov::Tensor data_tensor_0 = ov::test::utils::create_and_fill_tensor(ov::element::f16, shapes[0], data0);
        inputs.insert({model_inputs[0].get_node_shared_ptr(), data_tensor_0});
        // K
        shapes[1] = targetInputStaticShapes[1];
        ov::test::utils::InputGenerateData data1(0, 8, 32);
        ov::Tensor data_tensor_1 = ov::test::utils::create_and_fill_tensor(ov::element::f16, shapes[1], data1);
        inputs.insert({model_inputs[1].get_node_shared_ptr(), data_tensor_1});
        // V
        shapes[2] = targetInputStaticShapes[2];
        ov::test::utils::InputGenerateData data2(0, 8, 32);
        ov::Tensor data_tensor_2 = ov::test::utils::create_and_fill_tensor(ov::element::f16, shapes[2], data2);
        inputs.insert({model_inputs[2].get_node_shared_ptr(), data_tensor_2});
    }
    ov::test::utils::InputGenerateData attn_data(-1.0f, 2, 1);
    ov::test::utils::InputGenerateData scale_data(0.1f, 1, 10);
    if (!has_attn && has_scale) {
        shapes.push_back(ov::Shape{});
        ov::Tensor attn_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f16, shapes[3], attn_data);
        inputs.insert({model_inputs[3].get_node_shared_ptr(), attn_tensor});
        if (!is_scale_const) {
            shapes.push_back(ov::Shape{1});
            ov::Tensor scale_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f16, shapes[4], scale_data);
            inputs.insert({model_inputs[4].get_node_shared_ptr(), scale_tensor});
        }
    } else {
        int idx = 3;
        if (has_attn && !is_attn_const) {
            shapes.push_back(targetInputStaticShapes[3]);
            ov::Tensor attn_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f16, shapes[idx], attn_data);
            inputs.insert({model_inputs[idx++].get_node_shared_ptr(), attn_tensor});
        }
        if (has_scale && !is_scale_const) {
            shapes.push_back(ov::Shape{1});
            ov::Tensor scale_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f16, shapes[idx], scale_data);
            inputs.insert({model_inputs[idx].get_node_shared_ptr(), scale_tensor});
        }
    }
}

TEST_P(ScaledAttnLayerGPUTest, CompareWithRefs) {
    run();
}
const std::vector<std::vector<InputShape>> dynamic_shapes_3D {
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 80},
            {ov::Shape{16, 128, 80}, ov::Shape{16, 1, 80}, ov::Shape{2, 10, 80}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 80},
            {ov::Shape{16, 128, 80}, ov::Shape{16, 1, 80}, ov::Shape{2, 10, 80}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 80},
            {ov::Shape{16, 128, 80}, ov::Shape{16, 1, 80}, ov::Shape{2, 10, 80}}}
        },
        // attn shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, -1},
            {ov::Shape{1, 128, 128}, ov::Shape{1, 1, 1}, ov::Shape{1, 10, 10}}}
        },
    },
};

const std::vector<std::vector<int64_t>> disable_transpose{};
const std::vector<std::vector<int64_t>> transpose_all_3D{{1, 0, 2}, {1, 0, 2}, {1, 0, 2}};
const auto dynamic_shape_params_3D = testing::Combine(testing::Values(ov::element::f16 /*, ov::element::f32 */),
                                                      testing::ValuesIn(dynamic_shapes_3D),
                                                      testing::Values(false),
                                                      testing::Values(true, false),
                                                      testing::Values(false),
                                                      testing::Values(true, false),
                                                      testing::Values(false),
                                                      testing::ValuesIn({disable_transpose, transpose_all_3D}),
                                                      testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_ScaledAttnDynamic3D_GPU,
                         ScaledAttnLayerGPUTest,
                         dynamic_shape_params_3D,
                         ScaledAttnLayerGPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> static_shapes_3D{
    // static shapes
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{16, 128, 80},
            {ov::Shape{16, 128, 80}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{16, 128, 80},
            {ov::Shape{16, 128, 80}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{16, 128, 80},
            {ov::Shape{16, 128, 80}}}
        },
        // attn shape: [B, 128, -128, L0+L1]
        {ov::test::InputShape{ov::PartialShape{128, 128},
            {ov::Shape{128, 128}}}
        },
    },
};

const std::vector<std::vector<InputShape>> static_shapes_4D{
    // static shapes
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{1, 16, 128, 80},
            {ov::Shape{1, 16, 128, 80}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{1, 16, 128, 80},
            {ov::Shape{1, 16, 128, 80}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{1, 16, 128, 80},
            {ov::Shape{1, 16, 128, 80}}}
        },
        // attn shape: [B, 128, -128, L0+L1]
        {ov::test::InputShape{ov::PartialShape{128, 128},
            {ov::Shape{128, 128}}}
        },
    },
};

const auto static_shape_params_4D_sink = testing::Combine(testing::Values(ov::element::f16),
                                                  testing::ValuesIn(static_shapes_4D),
                                                  testing::Values(false),
                                                  testing::Values(true),
                                                  testing::Values(false),
                                                  testing::Values(true),
                                                  testing::Values(false),
                                                  testing::ValuesIn({disable_transpose}),
                                                  testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_ScaledAttnStatic4DSink_GPU_taylor,
                         ScaledAttnLayerGPUTest,
                         static_shape_params_4D_sink,
                         ScaledAttnLayerGPUTest::getTestCaseName);



const std::vector<std::vector<InputShape>> dynamic_shapes_4D {
    // normal case, shapes of q,k,v are same
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 128},
            {ov::Shape{1, 8, 100, 128}, ov::Shape{1, 8, 1, 128}, ov::Shape{2, 8, 10, 128}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 128},
            {ov::Shape{1, 8, 100, 128}, ov::Shape{1, 8, 1, 128}, ov::Shape{2, 8, 10, 128}}}
        },
        // v shape
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
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 100, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // v shape
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
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 5, -1, 128},
            {ov::Shape{2, 5, 100, 128}, ov::Shape{2, 5, 1, 128}, ov::Shape{2, 5, 387, 128}}}
        },
        // v shape
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
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, 128},
            {ov::Shape{1, 1, 100, 128}, ov::Shape{1, 1, 1, 128}, ov::Shape{2, 1, 10, 128}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, 128},
            {ov::Shape{1, 1, 100, 128}, ov::Shape{1, 1, 1, 128}, ov::Shape{2, 1, 10, 128}}}
        },
        // attn shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, -1},
            {ov::Shape{1, 8, 100, 100}, ov::Shape{1, 8, 1, 1}, ov::Shape{2, 8, 10, 10}}}
        },
    },
    // heads number of qkv is 1, attn mask: [B, H, L1, L0+L1]
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, 80},
            {ov::Shape{16, 1, 128, 80}, ov::Shape{16, 1, 1, 80}, ov::Shape{2, 1, 10, 80}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, 80},
            {ov::Shape{16, 1, 128, 80}, ov::Shape{16, 1, 1, 80}, ov::Shape{2, 1, 10, 80}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, 80},
            {ov::Shape{16, 1, 128, 80}, ov::Shape{16, 1, 1, 80}, ov::Shape{2, 1, 10, 80}}}
        },
        // attn shape
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, -1},
            {ov::Shape{1, 1, 128, 128}, ov::Shape{1, 1, 1, 1}, ov::Shape{2, 1, 10, 10}}}
        },
    },
    // large head size
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 256},
            {ov::Shape{1, 8, 7, 256}, ov::Shape{1, 8, 1, 256}, ov::Shape{2, 8, 10, 256}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 256},
            {ov::Shape{1, 8, 7, 256}, ov::Shape{1, 8, 1, 256}, ov::Shape{2, 8, 10, 256}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 256},
            {ov::Shape{1, 8, 7, 256}, ov::Shape{1, 8, 1, 256}, ov::Shape{2, 8, 10, 256}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, -1},
            {ov::Shape{1, 1, 7, 7}, ov::Shape{1, 1, 1, 1}, ov::Shape{2, 1, 10, 10}}}
        },
    },
    // head size not aligned to 16
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 72},
            {ov::Shape{1, 8, 100, 72}, ov::Shape{1, 8, 32, 72}, ov::Shape{1, 8, 1, 72}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 72},
            {ov::Shape{1, 8, 100, 72}, ov::Shape{1, 8, 32, 72}, ov::Shape{1, 8, 1, 72}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 72},
            {ov::Shape{1, 8, 100, 72}, ov::Shape{1, 8, 32, 72}, ov::Shape{1, 8, 1, 72}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, -1},
            {ov::Shape{1, 1, 100, 100}, ov::Shape{1, 1, 32, 32}, ov::Shape{1, 1, 1, 1}}}
        },
    },
    // different head size
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 128},
            {ov::Shape{1, 8, 100, 128}, ov::Shape{1, 8, 1, 128}, ov::Shape{2, 8, 10, 128}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 128},
            {ov::Shape{1, 8, 100, 128}, ov::Shape{1, 8, 1, 128}, ov::Shape{2, 8, 10, 128}}}
        },
        // v shape
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
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 5, -1, 128},
            {ov::Shape{2, 5, 100, 128}, ov::Shape{2, 5, 1, 128}, ov::Shape{2, 5, 387, 128}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{-1, 5, -1, 32},
            {ov::Shape{2, 5, 100, 32}, ov::Shape{2, 5, 1, 32}, ov::Shape{2, 5, 387, 32}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, -1},
            {ov::Shape{1, 1, 100, 100}, ov::Shape{1, 1, 1, 1}, ov::Shape{2, 1, 387, 387}}}
        },
    },
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 100, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 100, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 96},
            {ov::Shape{1, 8, 100, 96}, ov::Shape{1, 8, 1, 96}, ov::Shape{2, 8, 10, 96}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, -1},
            {ov::Shape{1, 1, 100, 100}, ov::Shape{1, 1, 1, 1}, ov::Shape{2, 1, 10, 10}}}
        },
    },
    // single token
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 20, -1, 64},
            {ov::Shape{1, 20, 1, 64}, ov::Shape{1, 20, 1, 64}, ov::Shape{2, 20, 1, 64}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 20, -1, 64},
            {ov::Shape{1, 20, 2, 64}, ov::Shape{1, 20, 10, 64}, ov::Shape{2, 20, 2, 64}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{-1, 20, -1, 64},
            {ov::Shape{1, 20, 2, 64}, ov::Shape{1, 20, 10, 64}, ov::Shape{2, 20, 2, 64}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, -1},
            {ov::Shape{1, 1, 1, 2}, ov::Shape{1, 1, 1, 10}, ov::Shape{2, 1, 1, 2}}}
        },
    },
    // 4D inputs, 2D mask
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 245, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 245, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 245, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{-1, -1},
            {ov::Shape{245, 245}, ov::Shape{1, 1}, ov::Shape{10, 10}}}
        },
    },
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 10, -1, 64},
            {ov::Shape{1, 10, 77, 64}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{-1, 10, -1, 64},
            {ov::Shape{1, 10, 77, 64}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{-1, 10, -1, 64},
            {ov::Shape{1, 10, 77, 64}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{77, 77},
            {ov::Shape{77, 77}}}
        },
    }
};

const std::vector<std::vector<int64_t>> transpose_value{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 2, 1, 3}};
const std::vector<std::vector<int64_t>> transpose_all_4D{{0, 2, 1, 3}, {0, 2, 1, 3}, {0, 2, 1, 3}};

const auto dynamic_shape_params_4D = testing::Combine(testing::Values(ov::element::f16 /*, ov::element::f32 */),
                                                   testing::ValuesIn(dynamic_shapes_4D),
                                                   testing::Values(true, false),
                                                   testing::Values(true, false),
                                                   testing::Values(true, false),
                                                   testing::Values(true, false),
                                                   testing::Values(true, false),
                                                   testing::ValuesIn({disable_transpose, transpose_value}),
                                                   testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_ScaledAttnDynamic4D_GPU,
                         ScaledAttnLayerGPUTest,
                         dynamic_shape_params_4D,
                         ScaledAttnLayerGPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> static_shapes{
    // static shapes
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 100, 128},
            {ov::Shape{1, 8, 100, 128}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 100, 128},
            {ov::Shape{1, 8, 100, 128}}}
        },
        // v shape
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
        // k shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 13, 128},
            {ov::Shape{1, 8, 13, 128}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 13, 128},
            {ov::Shape{1, 8, 13, 128}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{1, 1, 64, 13},
            {ov::Shape{1, 1, 64, 13}}}
        },
    },
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 100, 72},
            {ov::Shape{1, 8, 100, 72}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 100, 72},
            {ov::Shape{1, 8, 100, 72}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 100, 72},
            {ov::Shape{1, 8, 100, 72}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{1, 1, 100, 100},
            {ov::Shape{1, 1, 100, 100}}}
        },
    },
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 1, 72},
            {ov::Shape{1, 8, 1, 72}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 100, 72},
            {ov::Shape{1, 8, 100, 72}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 100, 72},
            {ov::Shape{1, 8, 100, 72}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{1, 1, 1, 100},
            {ov::Shape{1, 1, 1, 100}}}
        },
    },
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{1, 10, 77, 64},
            {ov::Shape{1, 10, 77, 64}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{1, 10, 77, 64},
            {ov::Shape{1, 10, 77, 64}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{1, 10, 77, 64},
            {ov::Shape{1, 10, 77, 64}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{77, 77},
            {ov::Shape{77, 77}}}
        },
    },
};

const auto static_shape_params = testing::Combine(testing::Values(ov::element::f16),
                                                  testing::ValuesIn(static_shapes),
                                                  testing::Values(true, false),
                                                  testing::Values(true, false),
                                                  testing::Values(false),
                                                  testing::Values(true, false),
                                                  testing::Values(false),
                                                  testing::ValuesIn({disable_transpose, transpose_all_4D}),
                                                  testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_ScaledAttnStatic_GPU,
                         ScaledAttnLayerGPUTest,
                         static_shape_params,
                         ScaledAttnLayerGPUTest::getTestCaseName);
} // namespace
