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
                   bool                              // has_scale
                   > ScaledAttnGPUTestParams;

class ScaledAttnLayerGPUTest : public testing::WithParamInterface<ScaledAttnGPUTestParams>,
                               virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ScaledAttnGPUTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    bool is_causal;
    bool has_attn;
    bool has_scale;
};

std::string ScaledAttnLayerGPUTest::getTestCaseName(const testing::TestParamInfo<ScaledAttnGPUTestParams>& obj) {
    ov::element::Type inType;
    std::vector<InputShape> inputShapes;
    bool is_causal;
    bool has_attn;
    bool has_scale;
    std::tie(inType, inputShapes, is_causal, has_attn, has_scale) = obj.param;

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

    return result.str();
}

void ScaledAttnLayerGPUTest::SetUp() {
    ov::element::Type inType;
    std::vector<InputShape> inputShapes;

    targetDevice = ov::test::utils::DEVICE_GPU;

    std::tie(inType, inputShapes, is_causal, has_attn, has_scale) = this->GetParam();

    init_input_shapes(inputShapes);
    ov::ParameterVector inputParams;
    // q, k, v
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]));
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
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[2]));
            inputParams.back()->set_friendly_name("attention_mask");
        }
        if (has_scale) {
            // scale：[1]
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{1}));
            inputParams.back()->set_friendly_name("scale");
        }
    }

    ov::OutputVector inputs;
    for (size_t i = 0; i < inputParams.size(); i++) {
        inputs.push_back(inputParams[i]);
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

void ScaledAttnLayerGPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    std::vector<ov::Shape> shapes(3);
    shapes[0] = targetInputStaticShapes[0];
    shapes[1] = targetInputStaticShapes[1];
    shapes[2] = targetInputStaticShapes[1];
    if (!has_attn && has_scale) {
        shapes.push_back(ov::Shape{});
        shapes.push_back(ov::Shape{1});
    } else {
        if (has_attn) {
            shapes.push_back(targetInputStaticShapes[2]);
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
    bool is_causal;
    bool has_attn;
    bool has_scale;
    std::tie(inType, inputShapes, is_causal, has_attn, has_scale) = this->GetParam();
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
    },
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
};

const auto params = testing::Combine(testing::Values(ov::element::f16 /*, ov::element::f32 */),
                                                 testing::ValuesIn(shapes),
                                                 testing::Values(true, false),
                                                 testing::Values(true, false),
                                                 testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(smoke_ScaledAttn_GPU,
                         ScaledAttnLayerGPUTest,
                         params,
                         ScaledAttnLayerGPUTest::getTestCaseName);

} // namespace
