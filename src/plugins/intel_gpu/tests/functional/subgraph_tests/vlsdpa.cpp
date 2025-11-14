// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/engine_configuration.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"

#include "openvino/opsets/opset13.hpp"
#include "ov_ops/vl_sdpa.hpp"

#include "openvino/util/log.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
// #include "../src/graph/impls/ocl/kernel_selector_helper.h"

//=================================================================================
// Transpose + SDPA + Transpose pattern fusion (TransposeVLSDPAMatcher)
// In this test case, SDPA is first transformed to VLSDPA under desired situations.
// Then the pattern is transformed to VLSDPA after pass of TransposeVLSDPAMatcher
// is applied.
//=================================================================================
namespace ov {
namespace test {
using namespace ov;
using namespace ov::opset13;
using namespace ov::intel_gpu;

using TransposeVLSDPATestParams = std::tuple<ElementType,
                                             ov::Dimension::value_type,     // num_head
                                             ov::Dimension::value_type,     // head_size
                                             std::vector<int32_t>>;         // cu_seqlens

class TransposeVLSDPATestOnGPU: public testing::WithParamInterface<TransposeVLSDPATestParams>,
                                virtual public test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeVLSDPATestParams>& obj) {
        ElementType inType;
        ov::Dimension::value_type num_head, head_size;
        std::vector<int32_t> cu_seqlens;

        std::tie(inType, num_head, head_size, cu_seqlens) = obj.param;

        std::ostringstream result;
        result << "device=(" << std::string(utils::DEVICE_GPU) << ")_";
        result << "num_head=(" << to_str(num_head) << ")_";
        result << "head_size=(" << to_str(head_size) << ")_";
        result << test::utils::vec2str<int32_t>({cu_seqlens}) << "_";
        result << "Prc=" << inType;
        return result.str();
    }

    static std::shared_ptr<Parameter> make_param(const PartialShape& pshape,
                                                element::Type element_type,
                                                const std::string& name) {
        auto param = std::make_shared<Parameter>(element_type, pshape);
        param->set_friendly_name(name);
        param->get_output_tensor(0).set_names({name});
        return param;
    }

    static bool check_vl_sdpa_transformations(const ov::CompiledModel& compiled_model) {
        const std::vector<std::string> target_names {"cu_seq_lens", "cu_window_seqlens"};

        bool exists = false;
        for (auto &input : compiled_model.inputs()) {
            const auto& names = input.get_names();

            for (const auto& target : target_names) {
                exists |= (names.find(target) != names.end());
            }
        }

        return exists;
    }

protected:
    void SetUp() override {
        ElementType inType;
        ov::Dimension::value_type num_head, head_size;
        std::vector<int32_t> cu_seqlens;
        std::tie(inType, num_head, head_size, cu_seqlens) = GetParam();

        targetDevice = test::utils::DEVICE_GPU;
        rel_threshold = 0.02f;
        abs_threshold = 0.02f;
        if (inType == ov::element::f32)
            configuration[ov::hint::inference_precision.name()] = ov::element::f32.get_type_name();

        assert(cu_seqlens.front() == 0);
        const auto cumsum = cu_seqlens.back();
        const auto L = static_cast<size_t>(cumsum);
        const auto H = static_cast<size_t>(num_head);
        const auto S = static_cast<size_t>(head_size);
        const std::vector<InputShape> inputShapes = {
            // q
            {PartialShape{-1, num_head, head_size}, {Shape{L, H, S}}},
            // k
            {PartialShape{-1, num_head, head_size}, {Shape{L, H, S}}},
            // v
            {PartialShape{-1, num_head, head_size}, {Shape{L, H, S}}},
            // cu_seqlen
            {PartialShape{-1}, {Shape{cu_seqlens.size()}}},
        };
        init_input_shapes(inputShapes);

        function = get_function(inType, num_head, head_size);
        functionRefs = function->clone();

        m_cu_seqlens = cu_seqlens;
    }

    std::shared_ptr<ov::Model> get_function(ov::element::Type inType, ov::Dimension::value_type num_head, ov::Dimension::value_type head_size) {
        auto q = make_param(PartialShape{ov::Dimension::dynamic(), num_head, head_size}, inType, "q");
        auto k = make_param(PartialShape{ov::Dimension::dynamic(), num_head, head_size}, inType, "k");
        auto v = make_param(PartialShape{ov::Dimension::dynamic(), num_head, head_size}, inType, "v");
        auto attn_mask = make_param(PartialShape{1, -1, -1}, inType, "attention_mask");

        auto transpose_q = std::make_shared<Transpose>(q, Constant::create(element::i64, Shape{3}, order_q));
        auto transpose_k = std::make_shared<Transpose>(k, Constant::create(element::i64, Shape{3}, order_k));
        auto transpose_v = std::make_shared<Transpose>(v, Constant::create(element::i64, Shape{3}, order_v));
        transpose_q->set_friendly_name("transpose_q");
        transpose_k->set_friendly_name("transpose_k");
        transpose_v->set_friendly_name("transpose_v");

        const auto casual = false;
        auto sdpa = std::make_shared<opset13::ScaledDotProductAttention>(transpose_q,
                                                                         transpose_k,
                                                                         transpose_v,
                                                                         attn_mask, casual);
        sdpa->set_friendly_name("sdpa");

        auto transpose_o = std::make_shared<Transpose>(sdpa, Constant::create(element::i64, Shape{3}, order_o));
        transpose_o->set_friendly_name("transpose_o");

        auto model = std::make_shared<Model>(transpose_o, ParameterVector{q, k, v, attn_mask});
        model->set_rt_info("QWenVL", "model_type_hint");        // request_vl_sdpa_transformations
        return model;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        inputs_ref.clear();
        const auto& funcInputs = compiledModel.inputs();
        for (size_t i = 0lu; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];

            // function: q,k,v,cu_seqlens v.s. functionRef: q,k,v,attn_mask
            if (i < 3) { // q, k, v
                auto tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                inputs.insert({std::const_pointer_cast<Node>(funcInput.get_node_shared_ptr()), tensor});
                inputs_ref.emplace_back(tensor);
            } else { // cu_seqlens
                auto attn_mask = get_attention_mask<ov::float16>(m_cu_seqlens, ov::element::f16);
                auto cu_seqlens = get_cu_seqlens(m_cu_seqlens);
                if (check_vl_sdpa_transformations(compiledModel)) {
                    inputs.insert({std::const_pointer_cast<Node>(funcInput.get_node_shared_ptr()), cu_seqlens});
                } else {
                    inputs.insert({std::const_pointer_cast<Node>(funcInput.get_node_shared_ptr()), attn_mask});
                }

                inputs_ref.emplace_back(attn_mask);
            }
        }
    }

    std::vector<ov::Tensor> calculate_refs() override {
        if (is_report_stages) {
            std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is started"<< std::endl;
        }
        auto start_time = std::chrono::system_clock::now();

        auto outputs = ov::test::utils::infer_on_template(functionRefs, inputs_ref);
        // auto outputs = infer_on_gpu(functionRefs, inputs_ref);

        if (is_report_stages) {
            auto end_time = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is finished successfully. Duration is " << duration.count() << "s" << std::endl;
        }
        return outputs;
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        ASSERT_EQ(expected.size(), function->get_results().size());
        const auto& results = function->get_results();
        for (size_t j = 0; j < results.size(); j++) {
            const auto result = results[j];
            for (size_t i = 0; i < result->get_input_size(); ++i) {
                utils::compare(expected[j], actual[j], abs_threshold, rel_threshold);
            }
        }
    }

private:
    ov::TensorVector infer_on_gpu(const std::shared_ptr<ov::Model>& model, const ov::TensorVector& input_tensors) {
        std::map<std::shared_ptr<ov::Node>, ov::Tensor> inputs;
        auto params = model->inputs();
        OPENVINO_ASSERT(params.size() == input_tensors.size());
        for (size_t i = 0; i < params.size(); i++) {
            inputs[params[i].get_node_shared_ptr()] = input_tensors[i];
        }
        return infer_on_gpu(model, inputs);
    }

    ov::TensorVector infer_on_gpu(const std::shared_ptr<ov::Model>& model,
                                  const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs) {
        auto core = ov::test::utils::PluginCache::get().core();

        auto compiled_model = core->compile_model(model,
                                                  ov::test::utils::DEVICE_GPU,
                                                  configuration);
        auto infer_request = compiled_model.create_infer_request();

        for (auto& input : inputs) {
            infer_request.set_tensor(input.first, input.second);
        }
        infer_request.infer();

        ov::TensorVector outputs;
        for (const auto& output : model->outputs()) {
            outputs.push_back(infer_request.get_tensor(output));
        }

        return outputs;
    }

    template <typename ET>
    ov::Tensor get_attention_mask(std::vector<int32_t> cu_seqlens, ov::element::Type_t inType) const {
        assert(cu_seqlens.front() == 0);
        int32_t cumsum = cu_seqlens.back();

        // Create attention mask for vision embeddings merger model
        size_t hidden_states_size = cumsum;
        ov::Tensor attention_mask{inType, {1, hidden_states_size, hidden_states_size}};
        ET* attention_mask_data = attention_mask.data<ET>();
        std::fill_n(attention_mask_data, attention_mask.get_size(), -std::numeric_limits<ET>::infinity());

        for (size_t i = 1; i < cu_seqlens.size(); ++i) {
            size_t start = cu_seqlens[i-1];
            size_t end = cu_seqlens[i];
            for (size_t row = start; row < end; ++row) {
                for (size_t col = start; col < end; ++col) {
                    attention_mask_data[row * hidden_states_size + col] = ET(0.0f);
                }
            }
        }
        return attention_mask;
    }

    ov::Tensor get_cu_seqlens(std::vector<int32_t> cu_seqlens) const {
        assert(cu_seqlens.front() == 0);
        ov::Tensor t_cu_seqlens = ov::Tensor(ov::element::i32, {cu_seqlens.size()});
        auto* ptr = static_cast<int32_t*>(t_cu_seqlens.data());
        for (size_t n = 0; n < cu_seqlens.size(); n++) {
            ptr[n] = cu_seqlens[n];
        }
        return t_cu_seqlens;
    }

    const std::vector<int64_t> order_q = {1, 0, 2};
    const std::vector<int64_t> order_k = {1, 0, 2};
    const std::vector<int64_t> order_v = {1, 0, 2};
    const std::vector<int64_t> order_o = {1, 0, 2};

    std::vector<int32_t> m_cu_seqlens;

    ov::TensorVector inputs_ref;
};


TEST_P(TransposeVLSDPATestOnGPU, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ElementType inType;
    ov::Dimension::value_type num_head, head_size;
    std::vector<int32_t> cu_seqlens;
    std::tie(inType, num_head, head_size, cu_seqlens) = GetParam();
    if (inType != ElementType::f16) // VLSDPA CM kernel supports half precision only
        GTEST_SKIP();

    run();
}

namespace {

// cu_seqlens starts from 0, ends with seqlen.
const std::vector<std::vector<int32_t>> input_cu_seqlens = {
        {0, 16},
        {0, 16, 32},
        {0, 64, 128, 192, 256}
};

INSTANTIATE_TEST_SUITE_P(smoke_TransposeVLSDPATest,
                         TransposeVLSDPATestOnGPU,
                         ::testing::Combine(::testing::Values(ov::element::f16),
                                            ::testing::Values(1, 2),   // num_heads
                                            ::testing::Values(16, 64, 128),       // head_size
                                            ::testing::ValuesIn(input_cu_seqlens)),
                         TransposeVLSDPATestOnGPU::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
