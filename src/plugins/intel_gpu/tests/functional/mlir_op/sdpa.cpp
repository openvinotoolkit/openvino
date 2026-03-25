// Copyright (C) 2018-2026 Intel Corporation
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

typedef std::tuple<ov::element::Type,                 // netPrecision
                   std::vector<InputShape>,           // shape
                   bool,                              // is_causal
                   bool,                              // has_attn
                   bool,                              // is_attn_const
                   bool,                              // has_scale
                   bool,                              // is_scale_const
                   std::vector<std::vector<int64_t>>, // input_transpose
                   bool                               // has_sink
                   > ScaledAttnGPUTestParams;

class ScaledAttnLayerGPUMlirTest : public testing::WithParamInterface<ScaledAttnGPUTestParams>,
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

std::string ScaledAttnLayerGPUMlirTest::getTestCaseName(const testing::TestParamInfo<ScaledAttnGPUTestParams>& obj) {
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

void ScaledAttnLayerGPUMlirTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_GPU;
    configuration[ov::intel_gpu::hint::enable_sdpa_optimization.name()] = false;

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
    } else if (has_sink && !has_attn) {
        // Add default mask when sink token exists and attention mask is not present
        auto attn_const = std::make_shared<ov::op::v0::Constant>(inType, ov::Shape{}, 0.0f);
        attn_const->set_friendly_name("attention_mask");
        inputParams_transpose.push_back(attn_const);
    }
    if (has_scale && is_scale_const) {
        auto scale_const = std::make_shared<ov::op::v0::Constant>(inType, ov::Shape({1}), 0.35f);
        scale_const->set_friendly_name("scale");
        inputParams_transpose.push_back(scale_const);
    } else if (has_sink && !has_scale) {
        // Add default scale when sink token exists and scale is not present
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
        ov::test::utils::InputGenerateData data(0, 5, 100);
        auto sink_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f16, ov::Shape{1, num_heads, 1, 1}, data);
        auto sink_const = std::make_shared<ov::op::v0::Constant>(sink_tensor);
        sink_const->set_friendly_name("sink");
        inputs.push_back(sink_const);
    }
    auto sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(inputs, is_causal);
    sdp->set_friendly_name("sdpa");

    auto output = std::make_shared<ov::op::v0::Result>(sdp->output(0));

    function = std::make_shared<ov::Model>(ov::OutputVector{output}, inputParams, "sdpa_model");

    functionRefs = function->clone();

    // Set friendly name on the SDPA node in the reference model
    for (const auto& node : functionRefs->get_ops()) {
        if (ov::as_type_ptr<ov::opset13::ScaledDotProductAttention>(node)) {
            node->set_friendly_name("decompose_me_sdpa");
            break;
        }
    }

    ov::pass::Manager manager;

    // Decompose ScaledDotProductAttention
    manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    manager.run_passes(functionRefs);

    auto it = std::find_if(inputShapes[1].second.begin(), inputShapes[1].second.end(), [&](const ov::Shape& shape){
        return shape[0] >= 128 || shape[2] >= 384 || shape[3] >= 128;
    });

    bool has_diff_head_size = inputShapes[1].first.begin()[3] != inputShapes[2].first.begin()[3];

    bool has_long_seq = it != inputShapes[1].second.end();

    if (inType == ov::element::f16) {
        if (has_sink || (has_diff_head_size && !has_scale)) {
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

void ScaledAttnLayerGPUMlirTest::transpose_prepare(std::vector<InputShape>& shapes,
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

void ScaledAttnLayerGPUMlirTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const auto& model_inputs = function->inputs();
    inputs.clear();
    std::vector<ov::Shape> shapes(3);
    {
        for (int i = 0; i < 3; ++i) {
            shapes[i] = targetInputStaticShapes[i];
            ov::test::utils::InputGenerateData data(-1, 1, 64);
            ov::Tensor data_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f16, shapes[i], data);
            inputs.insert({model_inputs[i].get_node_shared_ptr(), data_tensor});
        }
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

    // Print first 10 values of each input
    for (const auto& [node, tensor] : inputs) {
        size_t n = std::min<size_t>(tensor.get_size(), 10);
        std::cout << "Input \"" << node->get_friendly_name() << "\" shape=" << tensor.get_shape()
                  << " type=" << tensor.get_element_type() << " first " << n << " values: [";
        if (tensor.get_element_type() == ov::element::f16) {
            auto* data = reinterpret_cast<const ov::float16*>(tensor.data());
            for (size_t i = 0; i < n; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << static_cast<float>(data[i]);
            }
        } else if (tensor.get_element_type() == ov::element::f32) {
            auto* data = reinterpret_cast<const float*>(tensor.data());
            for (size_t i = 0; i < n; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << data[i];
            }
        }
        std::cout << "]" << std::endl;
    }
}

TEST_P(ScaledAttnLayerGPUMlirTest, CompareWithRefs) {
    run();
}

const std::vector<std::vector<int64_t>> disable_transpose{};
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

const auto static_shape_params_3D = testing::Combine(testing::Values(ov::element::f16),
                                                  testing::ValuesIn(static_shapes_3D),
                                                  testing::Values(false),  // is_causal
                                                  testing::Values(false, true),  // has_attn
                                                  testing::Values(false, true),  // is_attn_const
                                                  testing::Values(false, true),  // has_scale
                                                  testing::Values(/*false,*/ true),  // is_scale_const
                                                  testing::ValuesIn({disable_transpose}),
                                                  testing::Values(false));  // has_sink

INSTANTIATE_TEST_SUITE_P(smoke_ScaledAttnStatic3D_GPU,
                         ScaledAttnLayerGPUMlirTest,
                         static_shape_params_3D,
                         ScaledAttnLayerGPUMlirTest::getTestCaseName);

const std::vector<std::vector<InputShape>> static_shapes_3D_4_4096_64{
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{4, 4096, 64},
            {ov::Shape{4, 4096, 64}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{4, 4096, 64},
            {ov::Shape{4, 4096, 64}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{4, 4096, 64},
            {ov::Shape{4, 4096, 64}}}
        },
        // attn shape (unused when has_attn=false)
        {ov::test::InputShape{ov::PartialShape{4096, 4096},
            {ov::Shape{4096, 4096}}}
        },
    },
};

const auto static_shape_params_3D_4_4096_64 = testing::Combine(
    testing::Values(ov::element::f16),
    testing::ValuesIn(static_shapes_3D_4_4096_64),
    testing::Values(false),       // is_causal
    testing::Values(false),       // has_attn
    testing::Values(false),       // is_attn_const
    testing::Values(false),       // has_scale
    testing::Values(false),       // is_scale_const
    testing::Values(disable_transpose),
    testing::Values(false));      // has_sink

INSTANTIATE_TEST_SUITE_P(smoke_ScaledAttnStatic3D_4_4096_64_GPU,
                         ScaledAttnLayerGPUMlirTest,
                         static_shape_params_3D_4_4096_64,
                         ScaledAttnLayerGPUMlirTest::getTestCaseName);


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
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 128, 128},
            {ov::Shape{1, 8, 128, 128}}}
        },
        // k shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 128, 128},
            {ov::Shape{1, 8, 128, 128}}}
        },
        // v shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 128, 128},
            {ov::Shape{1, 8, 128, 128}}}
        },
        // attn shape: [B, 1, -1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{1, 1, 128, 128},
            {ov::Shape{1, 1, 128, 128}}}
        },
    },
};

const auto static_shape_params_4D = testing::Combine(testing::Values(ov::element::f16),
                                                  testing::ValuesIn(static_shapes_4D),
                                                  testing::Values(false), // is_causal
                                                  testing::Values(false, true), // has_attn
                                                  testing::Values(false, true), // is_attn_const
                                                  testing::Values(false, true), // has_scale
                                                  testing::Values(/*false,*/true), // is_scale_const
                                                  testing::ValuesIn({disable_transpose}),
                                                  testing::Values(false)); // has_sink

INSTANTIATE_TEST_SUITE_P(smoke_ScaledAttnStatic4D_GPU,
                         ScaledAttnLayerGPUMlirTest,
                         static_shape_params_4D,
                         ScaledAttnLayerGPUMlirTest::getTestCaseName);

} // namespace
