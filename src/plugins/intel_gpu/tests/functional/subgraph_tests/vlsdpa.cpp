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
                                             std::vector<int32_t>>;  // cu_seqlens

class TransposeVLSDPATestOnGPU: public testing::WithParamInterface<TransposeVLSDPATestParams>,
                                virtual public test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TransposeVLSDPATestParams> obj) {
        ElementType inType;
        std::vector<int32_t> cu_seqlens;

        std::tie(inType, cu_seqlens) = obj.param;

        std::ostringstream result;
        result << "device=(" << std::string(utils::DEVICE_GPU) << ")_";
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

    static bool check_vlsdpa_available() {
        ElementType inType;
        std::vector<int32_t> cu_seqlens;
        std::tie(inType, cu_seqlens) = GetParam();
        if (inType != ElementType::f16) // VLSDPA CM kernel supports half precision only
            return false;
        // VLSDPA CM optimized for Xe1/Xe2 architectures only
        // auto engine = cldnn::engine::create(cldnn::engine_types::ocl, cldnn::runtime_types::ocl);
        // const auto& info = engine->get_device_info();
        // if (!(info.arch >= cldnn::gpu_arch::xe_lp)) {
        //     return false;
        // }

        // ExecutionConfig config;
        // if (!check_cm_jit_support(*engine.get(), config)) {
        //     OPENVINO_WARN("You may miss SDPAToVLSDPA optimization for QWenVL model,"
        //                 "as CM environment is unavailable. Enable it by installing proper GPU driver and CM compiler.");
        //     return false;
        // }

        return true;
    }

protected:
    void SetUp() override {
        ElementType inType;
        std::vector<int32_t> cu_seqlens;
        std::tie(inType, cu_seqlens) = GetParam();

        targetDevice = test::utils::DEVICE_GPU;
        rel_threshold = 0.02f;
        abs_threshold = 0.02f;
        if (inType == ov::element::f32)
            configuration[ov::hint::inference_precision.name()] = inType;

        assert(cu_seqlens.front() == 0);
        const auto cumsum = cu_seqlens.back();
        const auto L = static_cast<size_t>(cumsum);
        const auto H = static_cast<size_t>(head_num);
        const auto S = static_cast<size_t>(head_size);
        const std::vector<InputShape> inputShapes = {
            // q
            {PartialShape{-1, head_num, head_size}, {Shape{L, H, S}}},
            // k
            {PartialShape{-1, head_num, head_size}, {Shape{L, H, S}}},
            // v
            {PartialShape{-1, head_num, head_size}, {Shape{L, H, S}}},
            // cu_seqlen
            {PartialShape{-1}, {Shape{cu_seqlens.size()}}},
        };
        init_input_shapes(inputShapes);

        function = get_function(inType);
        functionRefs = get_ref_function(inType);

        m_cu_seqlens = cu_seqlens;
    }

    std::shared_ptr<ov::Model> get_ref_function(ov::element::Type inType) {
        auto q = make_param(PartialShape{ov::Dimension::dynamic(), head_num, head_size}, inType, "q");
        auto k = make_param(PartialShape{ov::Dimension::dynamic(), head_num, head_size}, inType, "k");
        auto v = make_param(PartialShape{ov::Dimension::dynamic(), head_num, head_size}, inType, "v");
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

        return std::make_shared<Model>(transpose_o, ParameterVector{q, k, v, attn_mask});
    }

    std::shared_ptr<ov::Model> get_function(ov::element::Type inType) {
        auto q = make_param(PartialShape{ov::Dimension::dynamic(), head_num, head_size}, inType, "q");
        auto k = make_param(PartialShape{ov::Dimension::dynamic(), head_num, head_size}, inType, "k");
        auto v = make_param(PartialShape{ov::Dimension::dynamic(), head_num, head_size}, inType, "v");
        auto cuseq_mask = make_param(PartialShape{-1}, element::i32, "cu_seq_lens");

        auto vlsdpa = std::make_shared<ov::op::internal::VLSDPA>(OutputVector{q, k, v, cuseq_mask},
                                                                 order_q, order_k, order_v, order_o);

        return std::make_shared<ov::Model>(NodeVector{vlsdpa}, ParameterVector{q, k, v, cuseq_mask});
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0lu; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];

            if (i < 3) { // q, k, v
                auto tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                std::cout << " ========== " << i << " funcInput.get_element_type() = " << funcInput.get_element_type() << ", " << tensor.get_size() << std::endl;
                if (funcInput.get_element_type() == ElementType::f32)
                    std::fill_n(tensor.data<float>(), tensor.get_size(), 2.0f);
                if (funcInput.get_element_type() == ElementType::f16)
                    std::fill_n(tensor.data<ov::float16>(), tensor.get_size(), ov::float16(2.0f));
                inputs.insert({funcInput.get_node_shared_ptr(), tensor});
            } else { // cu_seqlens
                auto tensor = get_cu_seqlens(m_cu_seqlens);
                inputs.insert({funcInput.get_node_shared_ptr(), tensor});
            }
        }   
    }

    std::vector<ov::Tensor> calculate_refs() override {
        if (is_report_stages) {
            std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is started"<< std::endl;
        }
        auto start_time = std::chrono::system_clock::now();

        // update_ref_model();
        if (!convert_precisions.empty()) {
            pass::Manager manager;
            manager.register_pass<ov::pass::ConvertPrecision>(convert_precisions, type_to_fuse_map{}, false, false);
            manager.run_passes(functionRefs);
            functionRefs->validate_nodes_and_infer_types();
        }

        // function: q,k,v,cu_seqlens -> functionRef: q,k,v,attn_mask
        std::map<std::shared_ptr<ov::Node>, ov::Tensor> inputs_ref;
        const auto& anchor_params = function->get_parameters(); 
        for (const auto& param : functionRefs->get_parameters()) {
            auto in_idx = functionRefs->get_parameter_index(param);
            if (in_idx < 3) { // q, k, v
                inputs_ref[param] = inputs.at(anchor_params[in_idx]);
            } else { // attn_mask
                inputs_ref[param] = get_attention_mask<ov::float16>(m_cu_seqlens);
            }                          
        }

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
    ov::Tensor get_attention_mask(std::vector<int32_t> cu_seqlens) {
        assert(cu_seqlens.front() == 0);
        int32_t cumsum = cu_seqlens.back();

        const auto& input_attn_mask = functionRefs->input("attention_mask");

        // Create attention mask for vision embeddings merger model
        size_t hidden_states_size = cumsum;
        ov::Tensor attention_mask{input_attn_mask.get_element_type(), {1, hidden_states_size, hidden_states_size}};
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

    ov::Tensor get_cu_seqlens(std::vector<int32_t> cu_seqlens) {
        assert(cu_seqlens.front() == 0);
        ov::Tensor t_cu_seqlens = ov::Tensor(ov::element::i32, {cu_seqlens.size()});
        auto* ptr = static_cast<int32_t*>(t_cu_seqlens.data());
        for (size_t n = 0; n < cu_seqlens.size(); n++) {
            ptr[n] = cu_seqlens[n];
        }
        return t_cu_seqlens;
    }

    const ov::Dimension::value_type head_num = 1, head_size = 128;
    const std::vector<int64_t> order_q = {1, 0, 2};
    const std::vector<int64_t> order_k = {1, 0, 2};
    const std::vector<int64_t> order_v = {1, 0, 2};
    const std::vector<int64_t> order_o = {1, 0, 2};

    std::vector<int32_t> m_cu_seqlens;
};

TEST_P(TransposeVLSDPATestOnGPU, CompareWithRefs){
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    if (!check_vlsdpa_available())
        GTEST_SKIP();

    run();
}

namespace {

// cu_seqlens starts from 0, ends with seqlen.
const std::vector<std::vector<int32_t>> input_cu_seqlens = {
        {0, 16}/*,       // 1x448x448
        {0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192},   // 8x448x448
        {0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
        1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920,
        1984, 2048, 2112, 2176, 2240, 2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816,
        2880, 2944, 3008, 3072, 3136, 3200, 3264, 3328, 3392, 3456, 3520, 3584, 3648, 3712,
        3776, 3840, 3904, 3968, 4032, 4096, 4160, 4224, 4288, 4352, 4416, 4480, 4544, 4608,
        4672, 4736, 4800, 4864, 4928, 4992, 5056, 5120, 5184, 5248, 5312, 5376, 5440, 5504,
        5568, 5632, 5696, 5760, 5824, 5888, 5952, 6016, 6080, 6144, 6208, 6272, 6336, 6400,
        6464, 6528, 6592, 6656, 6720, 6784, 6848, 6912, 6976, 7040, 7104, 7168, 7232, 7296,
        7360, 7424, 7488, 7552, 7616, 7680, 7744, 7808, 7872, 7936, 8000, 8064, 8128, 8192}  // 8x448x448 window attention*/
};

INSTANTIATE_TEST_SUITE_P(smoke_TransposeVLSDPATest,
                         TransposeVLSDPATestOnGPU,
                         ::testing::Combine(::testing::Values(ov::element::f32, ov::element::f16),
                                            ::testing::ValuesIn(input_cu_seqlens)),
                         TransposeVLSDPATestOnGPU::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
