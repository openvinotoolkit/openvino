// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "internal_properties.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;

namespace ov {
namespace test {
using InputShapes = std::vector<InputShape>;
using PagedAttnTestParams = std::tuple<ElementType, InputShapes, bool, bool, bool, int32_t, ov::AnyMap, bool>;

class PagedAttnTestBase : public testing::WithParamInterface<PagedAttnTestParams>,
                          virtual public ov::test::SubgraphBaseTest,
                          public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttnTestParams>& obj) {
        const auto& [inType,
                     inputShapes,
                     extendBlockIndices,
                     enableXattn,
                     sinkInput,
                     slidingWindow,
                     additional_config,
                     addSharedReader] = obj.param;
        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")_";
        }
        result << "Prc=" << inType << "_";
        result << "ExtendBlockIndices=" << extendBlockIndices << "_";
        result << "EnableXattn=" << enableXattn << "_";
        result << "SinkInput=" << sinkInput << "_";
        result << "SlidingWindow=" << slidingWindow << "_";
        if (addSharedReader)
            result << "SharedKVCache=1_";
        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second.as<std::string>() << "_";
        }
        result << ")";

        return result.str();
    }
    static std::shared_ptr<ov::op::v0::Parameter> make_param(const PartialShape& pshape,
                                                             element::Type element_type,
                                                             const std::string& name) {
        auto param = std::make_shared<v0::Parameter>(element_type, pshape);
        param->set_friendly_name(name);
        param->get_output_tensor(0).set_names({name});
        return param;
    }

    std::shared_ptr<ov::Model> get_model(ov::element::Type data_type,
                                         bool enable_xattn,
                                         ov::Dimension::value_type head_size = 64,
                                         ov::Dimension::value_type head_num = 8,
                                         bool use_sink_input = true,
                                         int32_t sliding_window = 0,
                                         bool add_shared_reader = false) {
        // q [batch_in_tokens, head_num * head_size]
        // k [batch_in_tokens, head_num * head_size]
        // v [batch_in_tokens, head_num * head_size]
        auto q = make_param(PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, data_type, "q");
        auto k = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "k");
        auto v = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "v");
        auto key_cache = make_param(PartialShape{ov::Dimension::dynamic(), 32, ov::Dimension::dynamic()},
                                    ov::element::dynamic,
                                    "key_cache.0");
        auto value_cache = make_param(PartialShape{ov::Dimension::dynamic(), 32, ov::Dimension::dynamic()},
                                      ov::element::dynamic,
                                      "value_cache.0");

        enable_keep_const_precision(key_cache);
        enable_keep_const_precision(value_cache);
        
        auto past_lens = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "past_lens");
        auto subsequence_begins =
            make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "subsequence_begins");
        auto block_indices = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices");
        auto block_indices_begins =
            make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices_begins");
        float scale_value = 1.0 / std::sqrt(head_size);
        auto scale =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale_value});
        auto silding_windows =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{sliding_window});
        auto alibi_slopes = std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
        auto max_context_len =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, std::vector<float>{1024});
        auto score_aggregation_window =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        auto rotated_block_indices =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto rotation_deltas =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto rotation_trig_lut =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
        auto xattention_threshold =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
        if (enable_xattn) {
            xattention_threshold =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{1}, std::vector<float>{0.9f});
        }
        auto xattention_block_size =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{64});
        auto xattention_stride =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{8});
        // Create sink input as Constant (not Parameter) for testing
        // PagedAttentionExtension requires sink input to be Constant
        // Use shape [1, head_num, 1, 1] when use_sink_input=true, or empty shape [0] when false
        std::shared_ptr<ov::op::v0::Constant> sinks;
        if (use_sink_input) {
            sinks = std::static_pointer_cast<ov::op::v0::Constant>(
                ov::test::utils::make_constant(data_type, Shape{1, static_cast<size_t>(head_num), 1, 1}));
        } else {
            // Create empty sink (matching SDPA->PA transformation behavior when no sink)
            sinks = std::static_pointer_cast<ov::op::v0::Constant>(ov::test::utils::make_constant(data_type, Shape{0}));
        }

        auto adaptive_rkv_start_size =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        auto adaptive_rkv_evictable_sizes =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto adaptive_rkv_diversity_block_set_indices =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto adaptive_rkv_diversity_block_set_indices_begins =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto token_type_ids = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});

        auto qq_bias = std::make_shared<ov::op::v0::Constant>(ov::element::u8, Shape{0}, std::vector<uint8_t>{0});
        auto qq_bias_begins = std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        ParameterVector params =
            {q, k, v, key_cache, value_cache, past_lens, subsequence_begins, block_indices, block_indices_begins};
        OutputVector paged_attn_inputs = {q,
                                          k,
                                          v,
                                          key_cache,
                                          value_cache,
                                          past_lens,
                                          subsequence_begins,
                                          block_indices,
                                          block_indices_begins,
                                          scale,
                                          silding_windows,
                                          alibi_slopes,
                                          max_context_len,
                                          score_aggregation_window,
                                          rotated_block_indices,
                                          rotation_deltas,
                                          rotation_trig_lut,
                                          xattention_threshold,
                                          xattention_block_size,
                                          xattention_stride,
                                          sinks,
                                          adaptive_rkv_start_size,
                                          adaptive_rkv_evictable_sizes,
                                          adaptive_rkv_diversity_block_set_indices,
                                          adaptive_rkv_diversity_block_set_indices_begins,
                                          token_type_ids,
                                          qq_bias,
                                          qq_bias_begins};

        auto paged_attn = std::make_shared<op::PagedAttentionExtension>(paged_attn_inputs);

        paged_attn->get_rt_info()["num_k_heads"] = head_num;
        paged_attn->get_rt_info()["k_head_size"] = head_size;
        paged_attn->get_rt_info()["num_v_heads"] = head_num;
        paged_attn->get_rt_info()["v_head_size"] = head_size;

        if (add_shared_reader) {
            // Gemma 4-style architecture: the writer layer's output feeds into the
            // reader layer via a residual connection (hidden = q + PA1_output).
            auto hidden = std::make_shared<ov::op::v1::Add>(q, paged_attn->output(0));

            // PA2 K/V inputs: separate parameters filled with distinct values at runtime.
            //
            // In a real Gemma 4 reader layer there are no K/V linear projections, but the
            // PA op signature still requires K/V inputs. The SDPAToPA transformation wires
            // whatever the original graph had into these ports. When write_kv_cache=false,
            // the kernel ignores them entirely — it only reads from the shared cache.
            auto k_reader =
                make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "k_reader");
            auto v_reader =
                make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "v_reader");
            params.push_back(k_reader);
            params.push_back(v_reader);

            OutputVector pa2_inputs = paged_attn_inputs;
            pa2_inputs[0] = hidden;
            pa2_inputs[1] = k_reader;
            pa2_inputs[2] = v_reader;
            auto paged_attn_2 = std::make_shared<op::PagedAttentionExtension>(pa2_inputs, /*write_kv_cache=*/false);
            paged_attn_2->get_rt_info()["num_k_heads"] = head_num;
            paged_attn_2->get_rt_info()["k_head_size"] = head_size;
            paged_attn_2->get_rt_info()["num_v_heads"] = head_num;
            paged_attn_2->get_rt_info()["v_head_size"] = head_size;
            return std::make_shared<ov::Model>(OutputVector{paged_attn, paged_attn_2}, params);
        }

        return std::make_shared<ov::Model>(OutputVector{paged_attn}, params);
    }

    virtual std::shared_ptr<ov::Model> get_ref_model(ov::element::Type data_type,
                                                     ov::Dimension::value_type head_size = 64,
                                                     ov::Dimension::value_type head_num = 8,
                                                     bool use_sink_input = true,
                                                     bool add_shared_reader = false) {
        // q, k, v use L,B,H,S layout
        ov::PartialShape q_shape, kv_shape, past_shape, atten_mask_shape, scale_shape, sink_shape;
        ov::ParameterVector inputParams;
        past_shape = {-1, 1, head_num, head_size};
        q_shape = {-1, 1, static_cast<int64_t>(head_num), head_size};
        kv_shape = {-1, 1, head_num, head_size};
        atten_mask_shape = {1, head_num, -1, -1};
        scale_shape = {1};
        sink_shape = {1, head_num, 1, 1};

        auto q = make_param(q_shape, data_type, "q");
        auto k = make_param(kv_shape, data_type, "k");
        auto v = make_param(kv_shape, data_type, "v");
        auto atten_mask = make_param(atten_mask_shape, data_type, "atten_mask");
        auto scale = make_param(scale_shape, data_type, "scale");
        std::shared_ptr<ov::op::v0::Parameter> sink = nullptr;
        if (use_sink_input) {
            sink = make_param(sink_shape, data_type, "sink");
        }
        auto past_kv = make_param(past_shape, data_type, "past_kv");
        inputParams.push_back(q);
        inputParams.push_back(k);
        inputParams.push_back(v);
        inputParams.push_back(atten_mask);
        inputParams.push_back(scale);
        if (use_sink_input) {
            inputParams.push_back(sink);
        }
        inputParams.push_back(past_kv);

        // Get the correct index for past_kv (it's always the last parameter before beam_idx)
        size_t past_kv_idx = inputParams.size() - 1;

        auto var_k =
            std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_shape, data_type, "pastk"});
        auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[past_kv_idx], var_k);
        pastk->set_friendly_name("pastk_r");
        auto var_v =
            std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_shape, data_type, "pastv"});
        auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[past_kv_idx], var_v);
        pastv->set_friendly_name("pastv_r");
        std::vector<size_t> transposeOrder{1, 2, 0, 3};
        auto preOrder = op::v0::Constant::create(ov::element::i32, {4}, transposeOrder);
        std::shared_ptr<ov::Node> q_in = std::make_shared<ov::op::v1::Transpose>(inputParams[0], preOrder);

        auto concat_axis = transposeOrder[2];
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        beam_idx->set_friendly_name("beam_idx");
        inputParams.push_back(beam_idx);
        auto gatherK =
            std::make_shared<ov::op::v8::Gather>(pastk,
                                                 beam_idx,
                                                 op::v0::Constant::create(ov::element::i32, {1}, {transposeOrder[0]}));
        auto gatherV =
            std::make_shared<ov::op::v8::Gather>(pastv,
                                                 beam_idx,
                                                 op::v0::Constant::create(ov::element::i32, {1}, {transposeOrder[0]}));
        auto concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherK, inputParams[1]}, concat_axis);
        auto concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherV, inputParams[2]}, concat_axis);
        std::shared_ptr<ov::Node> k_in = concatK;
        std::shared_ptr<ov::Node> v_in = concatV;
        k_in = std::make_shared<ov::op::v1::Transpose>(k_in, preOrder);
        v_in = std::make_shared<ov::op::v1::Transpose>(v_in, preOrder);

        // Use SDPA constructor based on sink input parameter
        // Parameters order: q, k, v, atten_mask, scale, [sink], past_kv, beam_idx
        size_t atten_mask_idx = 3;
        size_t scale_idx = 4;
        size_t sink_idx = use_sink_input ? 5 : 0;

        std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdp;
        // For sliding window case, set causal=false because we provide explicit mask with sliding window logic
        // For normal case, set causal=true to let SDPA apply causal mask internally
        bool use_causal = (this->sliding_window == 0);

        if (use_sink_input) {
            // 7-parameter SDPA constructor with sink support
            // Parameters: query, key, value, attn_mask, scale, sink, causal
            sdp = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_in,
                                                                           k_in,
                                                                           v_in,
                                                                           inputParams[atten_mask_idx],
                                                                           inputParams[scale_idx],
                                                                           inputParams[sink_idx],
                                                                           use_causal);
        } else {
            // 6-parameter SDPA constructor without sink
            // Parameters: query, key, value, attn_mask, scale, causal
            sdp = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_in,
                                                                           k_in,
                                                                           v_in,
                                                                           inputParams[atten_mask_idx],
                                                                           inputParams[scale_idx],
                                                                           use_causal);
        }
        sdp->set_friendly_name("mha");
        auto pastk_assign = std::make_shared<ov::op::v6::Assign>(concatK, var_k);
        auto pastv_assign = std::make_shared<ov::op::v6::Assign>(concatV, var_v);
        pastk_assign->set_friendly_name("pastk_w");
        pastv_assign->set_friendly_name("pastv_w");
        auto get_reshape_order = [](const ov::PartialShape& qkv_shape,
                                    const std::vector<size_t>& transposeOrder) -> std::vector<size_t> {
            assert(transposeOrder.size() == 4);
            auto H = qkv_shape[transposeOrder[1]].get_length();
            auto S = qkv_shape[transposeOrder[3]].get_length();
            return std::vector<size_t>{0, static_cast<size_t>(H * S)};
        };
        const auto reshapeOrder = get_reshape_order(q_shape, transposeOrder);
        auto postOrder =
            ov::op::v0::Constant::create(ov::element::i32, {4}, std::vector<size_t>{2, 0, 1, 3});  // BHLS -> LBHS
        auto transposeSDP = std::make_shared<ov::op::v1::Transpose>(sdp, postOrder);

        auto constReshape = ov::op::v0::Constant::create(ov::element::i32, {2}, reshapeOrder);
        auto reshapeSDP =
            std::make_shared<ov::op::v1::Reshape>(transposeSDP,
                                                  constReshape,
                                                  true);  // use LBHS to better compare data between pa and sdpa
        SinkVector sinks{pastk_assign, pastv_assign};
        ov::OutputVector results{reshapeSDP};

        if (add_shared_reader) {
            // SDPA2 (reader): uses residual query (q + SDPA1_output), reads same KV cache.
            // transposeSDP is SDPA1 output in [L, B, H, S], same as inputParams[0] (q).
            auto hidden_4d = std::make_shared<ov::op::v1::Add>(inputParams[0], transposeSDP);
            auto hidden_query = std::make_shared<ov::op::v1::Transpose>(hidden_4d, preOrder);

            std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdp2;
            if (use_sink_input) {
                sdp2 = std::make_shared<ov::op::v13::ScaledDotProductAttention>(hidden_query,
                                                                                k_in,
                                                                                v_in,
                                                                                inputParams[atten_mask_idx],
                                                                                inputParams[scale_idx],
                                                                                inputParams[sink_idx],
                                                                                use_causal);
            } else {
                sdp2 = std::make_shared<ov::op::v13::ScaledDotProductAttention>(hidden_query,
                                                                                k_in,
                                                                                v_in,
                                                                                inputParams[atten_mask_idx],
                                                                                inputParams[scale_idx],
                                                                                use_causal);
            }
            sdp2->set_friendly_name("mha_reader");
            auto transposeSDP2 = std::make_shared<ov::op::v1::Transpose>(sdp2, postOrder);
            auto reshapeSDP2 = std::make_shared<ov::op::v1::Reshape>(transposeSDP2, constReshape, true);
            results.push_back(reshapeSDP2);
        }

        auto model = std::make_shared<Model>(results, sinks, inputParams, "sdpa_model");
        return model;
    }

    void SetUp() override {
        const auto& [inType,
                     inputShapes,
                     extendBlockIndices,
                     enableXattn,
                     sinkInput,
                     slidingWindow,
                     additional_config,
                     addSharedReader] = this->GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;
        rel_threshold = 1e-2f;
        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        if (inType == ElementType::bf16) {
            configuration[ov::hint::inference_precision.name()] = ov::element::bf16;
            rel_threshold = 0.01f;
        }

        configuration.insert(additional_config.begin(), additional_config.end());
        init_input_shapes(inputShapes);
        ov::ParameterVector inputParams;

        this->sliding_window = slidingWindow;
        function = get_model(inType, enableXattn, 64, 8, sinkInput, slidingWindow, addSharedReader);
        functionRefs = get_ref_model(inType, 64, 8, sinkInput, addSharedReader);
        targetDevice = ov::test::utils::DEVICE_CPU;
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        // Check if the reference model uses sink input by examining the number of parameters
        bool ref_model_uses_sink = (functionRefs->get_parameters().size() == 8);

        std::vector<ov::Shape> shapes;
        shapes.push_back(targetInputStaticShapes[0]);  // q
        shapes.push_back(targetInputStaticShapes[0]);  // k
        shapes.push_back(targetInputStaticShapes[0]);  // v
        // atten_mask shape: always rectangular [1, heads, q_len, total_kv_len]
        // total_kv_len = past_len_count + seq_len to cover all past and current KV tokens
        auto seq_len = targetInputStaticShapes[0][0];
        size_t total_kv_len = static_cast<size_t>(past_len_count) + seq_len;
        shapes.push_back({1, 8, seq_len, total_kv_len});  // atten_mask (rectangular)
        shapes.push_back({1});                            // scale

        if (ref_model_uses_sink) {
            shapes.push_back({1, 8, 1, 1});  // sink
        }

        shapes.push_back(targetInputStaticShapes[1]);  // past_kv
        // beam_idx shape: [batch]
        shapes.push_back({targetInputStaticShapes[0][1]});  // beam_idx

        SubgraphBaseTest::generate_inputs(shapes);
    }
    template <typename IT, typename T>
    static void strided_iota(IT first, size_t n, T value, T stride) {
        // Descending order: generate values from high to low
        // Generate descending values to simulate attention patterns where earlier tokens have higher scores.
        // This is useful for testing sliding window attention mechanisms where recent context is prioritized.
        for (size_t i = 0; i < n; i++) {
            const float idx = static_cast<float>(n - 1 - i);
            const T generated = value + stride * static_cast<T>(idx);
            *first++ = generated;
        }
    }
    virtual void generate(int idx,
                          const bool isPagedAttn,
                          const std::vector<ov::Shape>& targetInputStaticShapes,
                          bool extendBlockIndices,
                          bool use_sink_input = true) {
        inputs.clear();
        auto create_input = [this](std::shared_ptr<ov::op::v0::Parameter> param, ov::Shape shape, float val) {
            if (param->get_element_type() == ov::element::i32) {
                ov::Tensor t{ov::element::i32, shape};
                auto size = shape[0];
                auto* p = static_cast<int*>(t.data());
                auto start = static_cast<int>(val);
                for (size_t i = 0; i < size; i++) {
                    p[i] = (start + i) % size;
                }
                inputs.insert({param, t});
            } else if (param->get_element_type() == ov::element::f32) {
                ov::Tensor t{ov::element::f32, shape};
                strided_iota(static_cast<float*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            } else if (param->get_element_type() == ov::element::f16) {
                ov::Tensor t{ov::element::f16, shape};
                strided_iota(static_cast<ov::float16*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            } else {
                ASSERT_TRUE(param->get_element_type() == ov::element::bf16);
                ov::Tensor t{ov::element::bf16, shape};
                strided_iota(static_cast<ov::bfloat16*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            }
        };

        if (isPagedAttn) {
            auto qkv_shape = targetInputStaticShapes[0];
            // L, B, H, S -> L * B, H * S
            create_input(function->get_parameters()[0],
                         {qkv_shape[0] * qkv_shape[1], qkv_shape[2] * qkv_shape[3]},
                         idx + 1.0f);
            create_input(function->get_parameters()[1],
                         {qkv_shape[0] * qkv_shape[1], qkv_shape[2] * qkv_shape[3]},
                         idx + 2.0f);
            create_input(function->get_parameters()[2],
                         {qkv_shape[0] * qkv_shape[1], qkv_shape[2] * qkv_shape[3]},
                         idx + 3.0f);
            size_t batch_size_in_sequences = 1;
            // The test here simulates pagedAttn calcuation with 1 subsequence
            // idx = 0 means 1st token calculation, idx > 0 means 2nd token calculation
            int32_t total_blocks = intel_cpu::div_up(qkv_shape[0] * qkv_shape[1] + past_len_count, 32);
            ov::Tensor past_lens(ov::element::i32, {batch_size_in_sequences}),
                subsequence_begins(ov::element::i32, {batch_size_in_sequences + 1}),
                block_indices_begins(ov::element::i32, {batch_size_in_sequences + 1}),
                block_indices(ov::element::i32, {static_cast<size_t>(total_blocks == 0 ? 1 : total_blocks)});
            int32_t *past_lens_data = reinterpret_cast<int32_t*>(past_lens.data()),
                    *subsequence_begins_data = reinterpret_cast<int32_t*>(subsequence_begins.data()),
                    *block_indices_begins_data = reinterpret_cast<int32_t*>(block_indices_begins.data()),
                    *block_indices_data = reinterpret_cast<int32_t*>(block_indices.data());
            inputs.insert({function->get_parameters()[3], key_cache});
            inputs.insert({function->get_parameters()[4], value_cache});
            if (idx == 0) {
                past_lens_data[0] = 0;
                subsequence_begins_data[0] = 0;
                subsequence_begins_data[1] = targetInputStaticShapes[0][0];
                block_indices_begins_data[0] = 0;
                // test case here only has 1 block for prefill, but we allocate 2 blocks to simulate the vLLM case.
                // To test whether we have overflow in kernels.
                if (extendBlockIndices)
                    block_indices_begins_data[1] = 2;
                else
                    block_indices_begins_data[1] = total_blocks;
                for (int32_t i = 0; i < total_blocks; i++) {
                    block_indices_data[i] = i;
                }
            } else {
                past_lens_data[0] = past_len_count;
                subsequence_begins_data[0] = 0;
                subsequence_begins_data[1] = targetInputStaticShapes[0][0];
                block_indices_begins_data[0] = 0;
                block_indices_begins_data[1] = total_blocks;
                for (int32_t i = 0; i < total_blocks; i++) {
                    block_indices_data[i] = i;
                }
            }

            inputs.insert({function->get_parameters()[5], past_lens});
            inputs.insert({function->get_parameters()[6], subsequence_begins});
            inputs.insert({function->get_parameters()[7], block_indices});
            inputs.insert({function->get_parameters()[8], block_indices_begins});

            // Reader layer K/V inputs (params [9] and [10], only present when model
            // has a shared-cache reader PA). Values differ from writer's K/V so that
            // an incorrect write would visibly corrupt the cache.
            if (function->get_parameters().size() > 9) {
                // Fill k_reader/v_reader with zeros — maximally different from the
                // descending strided_iota data used for k/v (~13000 range).
                // If write_kv_cache=false is broken and PA2 writes these zeros to cache,
                // the attention output will differ drastically from the reference.
                auto reader_shape = ov::Shape{qkv_shape[0] * qkv_shape[1], qkv_shape[2] * qkv_shape[3]};
                for (size_t p = 9; p <= 10; p++) {
                    auto param = function->get_parameters()[p];
                    ov::Tensor t{param->get_element_type(), reader_shape};
                    memset(t.data(), 0, t.get_byte_size());
                    inputs.insert({param, t});
                }
            }

            // Note: sink is a Constant in the model, not a Parameter, so no need to provide input for it

            past_len_count += static_cast<int32_t>(qkv_shape[0]);

        } else {
            // Reference model for SDPA
            auto params = function->get_parameters();
            int param_idx = 0;

            create_input(params[param_idx++], targetInputStaticShapes[0], idx + 1.0f);  // q
            create_input(params[param_idx++], targetInputStaticShapes[0], idx + 2.0f);  // k
            create_input(params[param_idx++], targetInputStaticShapes[0], idx + 3.0f);  // v

            // atten_mask - always use rectangular mask [1, head_num, q_len, total_kv_len]
            auto mask_param = params[param_idx++];
            const size_t head_num = targetInputStaticShapes[0][2];
            const size_t q_len = targetInputStaticShapes[0][0];
            const size_t total_kv_len = static_cast<size_t>(past_len_count) + q_len;

            if (sliding_window > 0) {
                // Sliding window: rectangular mask with sliding window logic
                ov::Tensor mask_tensor(ov::element::f32, {1, head_num, q_len, total_kv_len});
                auto* mask_data = mask_tensor.data<float>();
                const float neg_inf = -std::numeric_limits<float>::infinity();
                const int32_t offset = -sliding_window;

                for (size_t h = 0; h < head_num; ++h) {
                    for (size_t q_pos = 0; q_pos < q_len; ++q_pos) {
                        const int32_t global_q_idx = past_len_count + static_cast<int32_t>(q_pos);
                        for (size_t kv_idx = 0; kv_idx < total_kv_len; ++kv_idx) {
                            const int32_t global_k_idx = static_cast<int32_t>(kv_idx);
                            const bool within_window = global_k_idx > global_q_idx + offset;
                            const bool causal = global_k_idx <= global_q_idx;
                            const bool allow = within_window && causal;
                            const size_t linear_idx = (h * q_len + q_pos) * total_kv_len + kv_idx;
                            mask_data[linear_idx] = allow ? 0.f : neg_inf;
                        }
                    }
                }
                inputs[mask_param] = mask_tensor;
                past_len_count += static_cast<int32_t>(q_len);
            } else {
                // Normal case: rectangular mask with all zeros (no masking needed, causal handled by SDPA)
                create_input(mask_param, {1, head_num, q_len, total_kv_len}, 0.0f);
                past_len_count += static_cast<int32_t>(q_len);
            }

            // scale - single value for scaling
            create_input(params[param_idx++], {1}, 1.0f / std::sqrt(64));  // scale

            // sink - only if model uses sink input
            if (use_sink_input) {
                create_input(params[param_idx++], {1, targetInputStaticShapes[0][2], 1, 1}, 0.1f);  // sink
            }

            // past_kv - For SDPA with ReadValue/Assign:
            // - Iteration 0: empty tensor [0,1,8,64], ReadValue uses this as initial value (empty)
            // - Iteration 1+: should be empty [0,1,8,64], ReadValue ignores this and uses Variable state
            // Note: We always pass empty tensor since ReadValue/Assign manages the actual KV cache
            auto past_kv_shape = targetInputStaticShapes[1];
            past_kv_shape[0] = 0;                                    // Always use empty past for ReadValue-based SDPA
            create_input(params[param_idx++], past_kv_shape, 0.0f);  // past_kv (empty)

            // beam_idx - shape matching batch dimension
            create_input(params[param_idx++], ov::Shape{targetInputStaticShapes[0][1]},
                         idx + 0.0f);  // beam_idx
        }
    }
    void prepare() {
        compile_model();
        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);
    }
    void reset() {
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }
    }
    void init_kv_cache(size_t block_nums) {
        for (const auto& input : compiledModel.inputs()) {
            for (auto& name : input.get_names()) {
                auto cache_precision = input.get_element_type();
                ov::PartialShape pshape;
                if (name.find("key_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    pshape[0] = block_nums;
                    key_cache = ov::Tensor(cache_precision, pshape.get_shape());
                    break;
                } else if (name.find("value_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    pshape[0] = block_nums;
                    value_cache = ov::Tensor(cache_precision, pshape.get_shape());
                    break;
                }
            }
        }
    }

    std::vector<ov::Tensor> run_pa_inference(bool extendBlockIndices, bool sinkInput) {
        std::vector<ov::Tensor> outputs;
        int idx = 0;
        for (auto&& shapes : targetStaticShapes) {
            generate(idx++, true, shapes, extendBlockIndices, sinkInput);
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            inferRequest.infer();
            for (size_t out_idx = 0; out_idx < compiledModel.outputs().size(); out_idx++) {
                auto tensor = inferRequest.get_output_tensor(out_idx);
                ov::Tensor copy{tensor.get_element_type(), tensor.get_shape()};
                tensor.copy_to(copy);
                outputs.push_back(copy);
            }
        }
        return outputs;
    }

    std::vector<size_t> transposeOrder;
    size_t keyGroupSize = 0;
    bool quantKeyByChannel = false;
    bool hasShapeOf;
    ov::Tensor key_cache;
    ov::Tensor value_cache;
    int32_t past_len_count = 0;
    int32_t sliding_window = 0;
};

class PagedAttnVSSDPATest : public PagedAttnTestBase {
public:
    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model, bool extendBlockIndices, bool sinkInput = true) {
        function = model;
        prepare();
        init_kv_cache(1024 / 32);
        return run_pa_inference(extendBlockIndices, sinkInput);
    }

    std::vector<ov::Tensor> run_ref_test(std::shared_ptr<ov::Model> model, bool sinkInput) {
        function = model;
        prepare();
        std::vector<ov::Tensor> outputs;
        int idx = 0;
        for (auto&& shapes : targetStaticShapes) {
            generate(idx++, false, shapes, false, sinkInput);
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            inferRequest.infer();
            for (size_t out_idx = 0; out_idx < compiledModel.outputs().size(); out_idx++) {
                auto tensor = inferRequest.get_output_tensor(out_idx);
                ov::Tensor copy{tensor.get_element_type(), tensor.get_shape()};
                tensor.copy_to(copy);
                outputs.push_back(copy);
            }
        }
        reset();
        return outputs;
    }
};

TEST_P(PagedAttnVSSDPATest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, inputShapes, extendBlockIndices, enableXattn, sinkInput, slidingWindow, additional_config,
                 addSharedReader] = this->GetParam();
    const bool isSageAttn =
        intel_cpu::contains_key_value(additional_config, {ov::intel_cpu::enable_sage_attn.name(), true});
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();
    if (isSageAttn && !(ov::with_cpu_x86_avx512_core_amx_int8() || CPUTestUtils::with_cpu_x86_avx2_vnni_2()))
        GTEST_SKIP();

    past_len_count = 0;

    // compare the logits from paged attn and sdpa
    auto actualOutputs = run_test(function, extendBlockIndices, sinkInput);
    // reference model doesn't support sage attention
    if (isSageAttn) {
        configuration[ov::intel_cpu::enable_sage_attn.name()] = false;
    }
    // Reset past_len_count before running reference test to ensure consistent mask generation
    past_len_count = 0;
    auto expectedOutputs = run_ref_test(functionRefs, sinkInput);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {
const std::vector<ov::AnyMap> additional_configs = {{{ov::intel_cpu::enable_sage_attn.name(), true}},
                                                    {{ov::intel_cpu::enable_sage_attn.name(), false}}};
const std::vector<InputShapes> inputShapeAndReorders = {  // greedy search
    {
        // L1, B, H, S
        {{-1, 1, 8, 64}, {{256, 1, 8, 64}, {1, 1, 8, 64}}},
        // B, L0, H, S
        {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {256, 1, 8, 64}}},
    }};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnVSSDPATest,
                         PagedAttnVSSDPATest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::bf16),
                                            ::testing::ValuesIn(inputShapeAndReorders),
                                            ::testing::Values(true, false),
                                            // TODO: Xattn should not direcctly compare with SDPA/decomposed Matmul
                                            // which not contain sparse logics
                                            ::testing::Values(true, false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::ValuesIn(additional_configs),
                                            ::testing::Values(false)),  // addSharedReader
                         PagedAttnTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnVSSDPATest_WithSlidingWindowAndSinks,
                         PagedAttnVSSDPATest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(inputShapeAndReorders),
                                            ::testing::Values(false),        // extendBlockIndices
                                            ::testing::Values(false),        // enableXattn
                                            ::testing::Values(true, false),  // sinkInput
                                            ::testing::Values(0, 8),         // sliding_window = 8
                                            ::testing::Values(ov::AnyMap{
                                                {ov::intel_cpu::enable_sage_attn.name(), false}}),
                                            ::testing::Values(false)),  // addSharedReader
                         PagedAttnTestBase::getTestCaseName);

// PA1(write=true) + PA2(write=false) sharing the same KV cache.
// Verifies that PA2 reads the cache populated by PA1 and produces matching output.
INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnSharedKVCache,
                         PagedAttnVSSDPATest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::bf16),
                                            ::testing::ValuesIn(inputShapeAndReorders),
                                            ::testing::Values(false),   // extendBlockIndices
                                            ::testing::Values(false),   // enableXattn
                                            ::testing::Values(false),   // sinkInput
                                            ::testing::Values(0),       // slidingWindow
                                            ::testing::Values(ov::AnyMap{
                                                {ov::intel_cpu::enable_sage_attn.name(), false}}),
                                            ::testing::Values(true)),   // addSharedReader
                         PagedAttnTestBase::getTestCaseName);
}  // namespace

class PagedAttnVSMatmulTest : public PagedAttnTestBase {
public:
    std::shared_ptr<ov::Model> get_ref_model(ov::element::Type data_type,
                                             ov::Dimension::value_type head_size = 64,
                                             ov::Dimension::value_type head_num = 8,
                                             bool use_sink_input = false,
                                             bool add_shared_reader = false) override {
        // PagedAttnVSMatmulTest reference model doesn't use sink input or shared reader
        (void)use_sink_input;  // Suppress unused parameter warning
        (void)add_shared_reader;
        // q, k, v use L,B,H,S layout
        ov::PartialShape q_shape, kv_shape, past_shape, atten_mask_shape, scale_shape;
        ov::ParameterVector inputParams;
        past_shape = {-1, 1, head_num, head_size};
        q_shape = {-1, 1, static_cast<int64_t>(head_num), head_size};
        kv_shape = {-1, 1, head_num, head_size};
        atten_mask_shape = {1, head_num, -1, -1};
        scale_shape = {1};

        auto q = make_param(q_shape, data_type, "q");
        auto k = make_param(kv_shape, data_type, "k");
        auto v = make_param(kv_shape, data_type, "v");
        auto atten_mask = make_param(atten_mask_shape, data_type, "atten_mask");
        auto scale = make_param(scale_shape, data_type, "scale");
        auto past_kv = make_param(past_shape, data_type, "past_kv");
        auto beam_idx = make_param(ov::PartialShape{-1}, ov::element::i32, "beam_idx");

        inputParams.push_back(q);
        inputParams.push_back(k);
        inputParams.push_back(v);
        inputParams.push_back(atten_mask);
        inputParams.push_back(scale);
        inputParams.push_back(past_kv);
        inputParams.push_back(beam_idx);
        auto var_k =
            std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_shape, data_type, "pastk"});
        auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[5], var_k);
        pastk->set_friendly_name("pastk_r");
        auto var_v =
            std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_shape, data_type, "pastv"});
        auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[5], var_v);
        pastv->set_friendly_name("pastv_r");
        std::vector<size_t> transposeOrder{1, 2, 0, 3};
        auto preOrder = op::v0::Constant::create(ov::element::i32, {4}, transposeOrder);
        std::shared_ptr<ov::Node> q_in = std::make_shared<ov::op::v1::Transpose>(inputParams[0], preOrder);
        auto concat_axis = transposeOrder[2];
        auto gatherK =
            std::make_shared<ov::op::v8::Gather>(pastk,
                                                 beam_idx,
                                                 op::v0::Constant::create(ov::element::i32, {1}, {transposeOrder[0]}));
        auto gatherV =
            std::make_shared<ov::op::v8::Gather>(pastv,
                                                 beam_idx,
                                                 op::v0::Constant::create(ov::element::i32, {1}, {transposeOrder[0]}));
        auto concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherK, inputParams[1]}, concat_axis);
        auto concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherV, inputParams[2]}, concat_axis);
        auto pastk_assign = std::make_shared<ov::op::v6::Assign>(concatK, var_k);
        auto pastv_assign = std::make_shared<ov::op::v6::Assign>(concatV, var_v);
        pastk_assign->set_friendly_name("pastk_w");
        pastv_assign->set_friendly_name("pastv_w");
        auto zero_1d_const = op::v0::Constant::create(ov::element::i32, {1}, {0});
        auto zero_scalar_const = op::v0::Constant::create(ov::element::i32, {}, {0});
        auto one_1d_const = op::v0::Constant::create(ov::element::i32, {1}, {1});
        auto one_scalar_const = op::v0::Constant::create(ov::element::i32, {}, {1});
        auto neg_2_1d_const = op::v0::Constant::create(ov::element::i32, {1}, {-2});
        auto neg_2_scalar_const = op::v0::Constant::create(ov::element::i32, {}, {-2});
        // mha structure
        auto ConvertLike_484 =
            std::make_shared<ov::op::v1::ConvertLike>(op::v0::Constant::create(ov::element::i32, {}, {1}), q_in);
        auto ConvertLike_491 =
            std::make_shared<ov::op::v1::ConvertLike>(op::v0::Constant::create(ov::element::i32, {}, {64}), q_in);
        auto Sqrt_492 = std::make_shared<ov::op::v0::Sqrt>(ConvertLike_491);
        auto Divide_493 = std::make_shared<ov::op::v1::Divide>(ConvertLike_484, Sqrt_492);
        auto Multiply_494 = std::make_shared<ov::op::v1::Multiply>(q_in, Divide_493);
        auto Transpose_442 = std::make_shared<ov::op::v1::Transpose>(concatK, preOrder);
        auto ShapeOf_479 = std::make_shared<ov::op::v3::ShapeOf>(Transpose_442, ov::element::i32);
        auto ShapeOf_495 = std::make_shared<ov::op::v3::ShapeOf>(ShapeOf_479, ov::element::i32);
        auto Add_497 =
            std::make_shared<ov::op::v1::Add>(ShapeOf_495, op::v0::Constant::create(ov::element::i32, {1}, {-2}));
        auto Squeeze_500 = std::make_shared<ov::op::v0::Squeeze>(Add_497, zero_1d_const);
        auto Range_501 =
            std::make_shared<ov::op::v4::Range>(zero_scalar_const, Squeeze_500, one_scalar_const, ov::element::i32);
        auto Add_496 =
            std::make_shared<ov::op::v1::Add>(ShapeOf_495, op::v0::Constant::create(ov::element::i32, {1}, {-1}));
        auto Concat_505 = std::make_shared<ov::op::v0::Concat>(OutputVector{Range_501, Add_496, Add_497}, 0);
        auto Transpose_506 = std::make_shared<ov::op::v1::Transpose>(Transpose_442, Concat_505);
        auto MatMul_510 = std::make_shared<ov::op::v0::MatMul>(Multiply_494, Transpose_506, false, false);
        // k_len
        auto Gather_517 = std::make_shared<ov::op::v8::Gather>(ShapeOf_479, neg_2_scalar_const, zero_scalar_const, 0);
        auto Range_531 =
            std::make_shared<ov::op::v4::Range>(zero_scalar_const, Gather_517, one_scalar_const, ov::element::i32);
        auto Unsqueeze_532 = std::make_shared<ov::op::v0::Unsqueeze>(Range_531, zero_scalar_const);
        auto ShapeOf_478 = std::make_shared<ov::op::v3::ShapeOf>(q_in, ov::element::i32);
        // q_len
        auto q_len = std::make_shared<ov::op::v8::Gather>(ShapeOf_478, neg_2_scalar_const, zero_scalar_const, 0);
        // past_len
        auto shape_past_len = std::make_shared<ov::op::v3::ShapeOf>(past_kv, ov::element::i32);
        auto past_len = std::make_shared<ov::op::v8::Gather>(shape_past_len, zero_scalar_const, zero_scalar_const, 0);
        auto total_len = std::make_shared<ov::op::v1::Add>(q_len, past_len);
        auto Range_534 =
            std::make_shared<ov::op::v4::Range>(zero_scalar_const, q_len, one_scalar_const, ov::element::i32);
        auto add_past_len = std::make_shared<ov::op::v1::Add>(Range_534, past_len);
        auto Unsqueeze_597 = std::make_shared<ov::op::v0::Unsqueeze>(add_past_len, one_scalar_const);
        auto GreaterEqual_598 = std::make_shared<ov::op::v1::Greater>(Unsqueeze_532, Unsqueeze_597);
        auto Unsqueeze_521 = std::make_shared<ov::op::v0::Unsqueeze>(q_len, zero_scalar_const);
        auto Unsqueeze_520 = std::make_shared<ov::op::v0::Unsqueeze>(Gather_517, zero_scalar_const);
        auto Concat_522 = std::make_shared<ov::op::v0::Concat>(OutputVector{Unsqueeze_521, Unsqueeze_520}, 0);
        auto Constant_523 = op::v0::Constant::create(element::u8, ov::Shape({}), {0});
        float negative_inf = -INFINITY;
        auto ConvertLike_511 =
            std::make_shared<ov::op::v1::ConvertLike>(v0::Constant::create(ov::element::f32, Shape{}, {negative_inf}),
                                                      MatMul_510);
        // mask
        auto Broadcast_524 = std::make_shared<ov::op::v1::Broadcast>(ConvertLike_511,
                                                                     Concat_522,
                                                                     Constant_523,
                                                                     AutoBroadcastSpec(AutoBroadcastType::NUMPY));
        auto ConvertLike_485 = std::make_shared<ov::op::v1::ConvertLike>(zero_scalar_const, q_in);
        auto Select_599 = std::make_shared<ov::op::v1::Select>(GreaterEqual_598, Broadcast_524, ConvertLike_485);
        auto Add_600 = std::make_shared<ov::op::v1::Add>(MatMul_510, Select_599);

        // softmax
        auto Softmax_9787 = std::make_shared<ov::op::v8::Softmax>(Add_600, -1);
        auto Transpose_443 = std::make_shared<ov::op::v1::Transpose>(concatV, preOrder);
        auto mha = std::make_shared<ov::op::v0::MatMul>(Softmax_9787, Transpose_443, false, false);
        auto Transpose_9634 = std::make_shared<ov::op::v1::Transpose>(
            mha,
            op::v0::Constant::create(ov::element::i32, {4}, std::vector<int32_t>{2, 0, 1, 3}));
        auto Reshape_9636 = std::make_shared<ov::op::v1::Reshape>(
            Transpose_9634,
            op::v0::Constant::create(ov::element::i32, {2}, std::vector<int32_t>{0, 512}),
            true);
        SinkVector sinks{pastk_assign, pastv_assign};
        ov::OutputVector results{Reshape_9636};
        auto model = std::make_shared<Model>(results, sinks, inputParams, "sdpa_model");
        return model;
    }

    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model,
                                     bool extendBlockIndices,
                                     bool sinkInput = false) {
        (void)sinkInput;  // Suppress unused parameter warning
        configuration[ov::hint::kv_cache_precision.name()] = ov::element::f16;
        function = model;
        prepare();
        init_kv_cache(4);
        return run_pa_inference(extendBlockIndices, false);
    }

    std::vector<ov::Tensor> run_ref_test(std::shared_ptr<ov::Model> model) {
        function = model;
        prepare();
        std::vector<ov::Tensor> outputs;
        int idx = 0;
        for (auto&& shapes : targetStaticShapes) {
            generate(idx++, false, shapes, false, false);
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            inferRequest.infer();
            auto outputTensor = inferRequest.get_output_tensor(0);
            ov::Tensor copy{outputTensor.get_element_type(), outputTensor.get_shape()};
            outputTensor.copy_to(copy);
            outputs.push_back(copy);
        }
        reset();
        return outputs;
    }
};

TEST_P(PagedAttnVSMatmulTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType,
                 inputShapes,
                 extendBlockIndices,
                 enableXattn,
                 sinkInput,
                 slidingWindow,
                 additional_config,
                 addSharedReader] = this->GetParam();
    ASSERT_FALSE(addSharedReader) << "PagedAttnVSMatmulTest does not support shared KV-cache (addSharedReader=true)";
    const bool isSageAttn =
        intel_cpu::contains_key_value(additional_config, {ov::intel_cpu::enable_sage_attn.name(), true});
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();
    if (isSageAttn && !(ov::with_cpu_x86_avx512_core_amx_int8() || CPUTestUtils::with_cpu_x86_avx2_vnni_2()))
        GTEST_SKIP();
    // compare the logits from paged attn and sdpa
    auto actualOutputs = run_test(function, extendBlockIndices, false);
    // reference model doesn't support sage attention, disable it
    if (isSageAttn) {
        configuration[ov::intel_cpu::enable_sage_attn.name()] = false;
    }
    auto expectedOutputs = run_ref_test(functionRefs);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {

const std::vector<InputShapes> inputShapes = {  // greedy search
    {
        // L1, B, H, S
        {{-1, 1, 8, 64}, {{10, 1, 8, 64}, {1, 1, 8, 64}}},
        // B, L0, H, S
        {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {10, 1, 8, 64}}},
    }};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnVSMatmulTest,
                         PagedAttnVSMatmulTest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::f16),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(true, false),
                                            ::testing::Values(true, false),
                                            ::testing::Values(false),  // sinkInput = false
                                            ::testing::Values(0),      // sliding_window = 0
                                            ::testing::ValuesIn(additional_configs),
                                            ::testing::Values(false)),  // addSharedReader
                         PagedAttnTestBase::getTestCaseName);
}  // namespace

// Regression test: executor cache collision with mixed head_size.
// Two PA nodes with head_size=256 and head_size=512 in ONE compiled model.
// Without the fix, PA2 reuses BRGEMM kernels configured for PA1's head_size,
// producing garbage. Requires f16 KV cache to trigger the BRGEMM code path.
class PagedAttnCacheCollisionTest : public PagedAttnTestBase {
public:
    static constexpr int64_t hs2_val = 512;

    void SetUp() override {
        const auto& [inType, inputShapes, extendBlockIndices, enableXattn, sinkInput,
                     slidingWindow, additional_config, addSharedReader] = this->GetParam();
        (void)enableXattn; (void)addSharedReader; (void)sinkInput; (void)slidingWindow;
        targetDevice = ov::test::utils::DEVICE_CPU;
        rel_threshold = 0.1f;
        abs_threshold = 0.05f;
        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        configuration.insert(additional_config.begin(), additional_config.end());
        init_input_shapes(inputShapes);
        this->sliding_window = 0;

        auto hs1 = static_cast<int64_t>(targetStaticShapes[0][0][3]);
        auto head_num = static_cast<int64_t>(targetStaticShapes[0][0][2]);

        function = get_mixed_head_model(inType, hs1, head_num);
        functionRefs = get_ref_model(inType, hs2_val, head_num, false, false);

        targetStaticShapes2_.reserve(targetStaticShapes.size());
        for (const auto& step : targetStaticShapes) {
            auto shape2 = step;
            shape2[0][3] = static_cast<size_t>(hs2_val);
            shape2[1][3] = static_cast<size_t>(hs2_val);
            targetStaticShapes2_.push_back(shape2);
        }
    }

    std::shared_ptr<ov::Model> get_mixed_head_model(ov::element::Type data_type, int64_t hs1, int64_t hn) {
        auto q1 = make_param(PartialShape{Dimension::dynamic(), Dimension::dynamic()}, data_type, "q1");
        auto k1 = make_param(PartialShape{Dimension::dynamic(), hn * hs1}, data_type, "k1");
        auto v1 = make_param(PartialShape{Dimension::dynamic(), hn * hs1}, data_type, "v1");
        auto kc1 = make_param(PartialShape{Dimension::dynamic(), 32, Dimension::dynamic()},
                              element::dynamic, "key_cache.0");
        auto vc1 = make_param(PartialShape{Dimension::dynamic(), 32, Dimension::dynamic()},
                              element::dynamic, "value_cache.0");
        enable_keep_const_precision(kc1);
        enable_keep_const_precision(vc1);

        auto q2 = make_param(PartialShape{Dimension::dynamic(), Dimension::dynamic()}, data_type, "q2");
        auto k2 = make_param(PartialShape{Dimension::dynamic(), hn * hs2_val}, data_type, "k2");
        auto v2 = make_param(PartialShape{Dimension::dynamic(), hn * hs2_val}, data_type, "v2");
        auto kc2 = make_param(PartialShape{Dimension::dynamic(), 32, Dimension::dynamic()},
                              element::dynamic, "key_cache.1");
        auto vc2 = make_param(PartialShape{Dimension::dynamic(), 32, Dimension::dynamic()},
                              element::dynamic, "value_cache.1");
        enable_keep_const_precision(kc2);
        enable_keep_const_precision(vc2);

        auto past_lens = make_param(PartialShape{Dimension::dynamic()}, element::i32, "past_lens");
        auto subseq = make_param(PartialShape{Dimension::dynamic()}, element::i32, "subsequence_begins");
        auto blk_idx = make_param(PartialShape{Dimension::dynamic()}, element::i32, "block_indices");
        auto blk_begins = make_param(PartialShape{Dimension::dynamic()}, element::i32, "block_indices_begins");

        auto make_consts = [](int64_t hs) {
            float sv = 1.0f / std::sqrt(static_cast<float>(hs));
            OutputVector c;
            c.push_back(v0::Constant::create(element::f32, Shape{}, {sv}));
            c.push_back(v0::Constant::create(element::i32, Shape{}, {0}));
            c.push_back(v0::Constant::create(element::f32, Shape{0}, std::vector<float>{}));
            c.push_back(v0::Constant::create(element::i32, Shape{}, {1024}));
            c.push_back(v0::Constant::create(element::i32, Shape{}, {0}));
            c.push_back(v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{0}));
            c.push_back(v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{0}));
            c.push_back(v0::Constant::create(element::f32, Shape{0}, std::vector<float>{0}));
            c.push_back(v0::Constant::create(element::f32, Shape{0}, std::vector<float>{0}));
            c.push_back(v0::Constant::create(element::i32, Shape{}, {64}));
            c.push_back(v0::Constant::create(element::i32, Shape{}, {8}));
            c.push_back(v0::Constant::create(element::f32, Shape{0}, std::vector<float>{0}));
            c.push_back(v0::Constant::create(element::i32, Shape{}, {0}));
            c.push_back(v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{0}));
            c.push_back(v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{0}));
            c.push_back(v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{0}));
            c.push_back(v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{}));
            c.push_back(v0::Constant::create(element::u8, Shape{0}, std::vector<uint8_t>{0}));
            c.push_back(v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{0}));
            return c;
        };

        auto consts1 = make_consts(hs1);
        OutputVector pa1_inputs = {q1, k1, v1, kc1, vc1, past_lens, subseq, blk_idx, blk_begins};
        pa1_inputs.insert(pa1_inputs.end(), consts1.begin(), consts1.end());
        auto pa1 = std::make_shared<PagedAttentionExtension>(pa1_inputs);
        pa1->get_rt_info()["num_k_heads"] = static_cast<size_t>(hn);
        pa1->get_rt_info()["k_head_size"] = static_cast<size_t>(hs1);
        pa1->get_rt_info()["num_v_heads"] = static_cast<size_t>(hn);
        pa1->get_rt_info()["v_head_size"] = static_cast<size_t>(hs1);

        auto residual = std::make_shared<v1::Add>(q1, pa1->output(0));

        auto consts2 = make_consts(hs2_val);
        OutputVector pa2_inputs = {q2, k2, v2, kc2, vc2, past_lens, subseq, blk_idx, blk_begins};
        pa2_inputs.insert(pa2_inputs.end(), consts2.begin(), consts2.end());
        auto pa2 = std::make_shared<PagedAttentionExtension>(pa2_inputs);
        pa2->get_rt_info()["num_k_heads"] = static_cast<size_t>(hn);
        pa2->get_rt_info()["k_head_size"] = static_cast<size_t>(hs2_val);
        pa2->get_rt_info()["num_v_heads"] = static_cast<size_t>(hn);
        pa2->get_rt_info()["v_head_size"] = static_cast<size_t>(hs2_val);

        ParameterVector params = {q1, k1, v1, kc1, vc1, q2, k2, v2, kc2, vc2,
                                  past_lens, subseq, blk_idx, blk_begins};
        return std::make_shared<Model>(OutputVector{residual, pa2->output(0)}, params);
    }

    void generate(int idx,
                  const bool isPagedAttn,
                  const std::vector<ov::Shape>& targetInputStaticShapes,
                  bool extendBlockIndices,
                  bool use_sink_input = true) override {
        if (!isPagedAttn) {
            PagedAttnTestBase::generate(idx, false, targetInputStaticShapes, extendBlockIndices, use_sink_input);
            return;
        }

        inputs.clear();
        auto fill_tensor = [](ov::Tensor& t, float val) {
            auto* p = t.data<float>();
            for (size_t i = 0; i < t.get_size(); i++)
                p[i] = val + 0.1f * static_cast<float>(t.get_size() - 1 - i);
        };

        auto params = function->get_parameters();
        auto seq_len = targetInputStaticShapes[0][0];
        size_t batch = targetInputStaticShapes[0][1];
        size_t tokens = seq_len * batch;
        size_t hs1 = static_cast<size_t>(targetStaticShapes[0][0][3]);
        size_t hn = targetStaticShapes[0][0][2];

        // PA1: q1, k1, v1
        ov::Tensor tq1(element::f32, {tokens, hn * hs1}); fill_tensor(tq1, idx + 10.0f);
        ov::Tensor tk1(element::f32, {tokens, hn * hs1}); fill_tensor(tk1, idx + 11.0f);
        ov::Tensor tv1(element::f32, {tokens, hn * hs1}); fill_tensor(tv1, idx + 12.0f);
        inputs.insert({params[0], tq1});
        inputs.insert({params[1], tk1});
        inputs.insert({params[2], tv1});
        inputs.insert({params[3], key_cache});
        inputs.insert({params[4], value_cache});

        // PA2: q2, k2, v2 — same offsets as SDPA reference
        size_t hs2 = static_cast<size_t>(hs2_val);
        ov::Tensor tq2(element::f32, {tokens, hn * hs2}); fill_tensor(tq2, idx + 1.0f);
        ov::Tensor tk2(element::f32, {tokens, hn * hs2}); fill_tensor(tk2, idx + 2.0f);
        ov::Tensor tv2(element::f32, {tokens, hn * hs2}); fill_tensor(tv2, idx + 3.0f);
        inputs.insert({params[5], tq2});
        inputs.insert({params[6], tk2});
        inputs.insert({params[7], tv2});
        inputs.insert({params[8], key_cache_1});
        inputs.insert({params[9], value_cache_1});

        // Shared block management
        int32_t total_blocks = intel_cpu::div_up(static_cast<int32_t>(tokens) + past_len_count, 32);
        ov::Tensor t_past(element::i32, {1});
        ov::Tensor t_sub(element::i32, {2});
        ov::Tensor t_bi(element::i32, {static_cast<size_t>(total_blocks == 0 ? 1 : total_blocks)});
        ov::Tensor t_bib(element::i32, {2});

        t_past.data<int32_t>()[0] = (idx == 0) ? 0 : past_len_count;
        t_sub.data<int32_t>()[0] = 0;
        t_sub.data<int32_t>()[1] = static_cast<int32_t>(tokens);
        t_bib.data<int32_t>()[0] = 0;
        t_bib.data<int32_t>()[1] = total_blocks;
        for (int32_t i = 0; i < total_blocks; i++)
            t_bi.data<int32_t>()[i] = i;

        inputs.insert({params[10], t_past});
        inputs.insert({params[11], t_sub});
        inputs.insert({params[12], t_bi});
        inputs.insert({params[13], t_bib});

        past_len_count += static_cast<int32_t>(tokens);
    }

    void init_all_kv_caches(size_t block_nums) {
        for (const auto& input : compiledModel.inputs()) {
            for (const auto& name : input.get_names()) {
                auto prec = input.get_element_type();
                auto ps = input.get_partial_shape();
                if (name == "key_cache.0") {
                    ps[0] = block_nums;
                    key_cache = ov::Tensor(prec, ps.get_shape());
                    std::memset(key_cache.data(), 0, key_cache.get_byte_size());
                } else if (name == "value_cache.0") {
                    ps[0] = block_nums;
                    value_cache = ov::Tensor(prec, ps.get_shape());
                    std::memset(value_cache.data(), 0, value_cache.get_byte_size());
                } else if (name == "key_cache.1") {
                    ps[0] = block_nums;
                    key_cache_1 = ov::Tensor(prec, ps.get_shape());
                    std::memset(key_cache_1.data(), 0, key_cache_1.get_byte_size());
                } else if (name == "value_cache.1") {
                    ps[0] = block_nums;
                    value_cache_1 = ov::Tensor(prec, ps.get_shape());
                    std::memset(value_cache_1.data(), 0, value_cache_1.get_byte_size());
                }
            }
        }
    }

    ov::Tensor key_cache_1;
    ov::Tensor value_cache_1;
    std::vector<std::vector<ov::Shape>> targetStaticShapes2_;
};

TEST_P(PagedAttnCacheCollisionTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    // Run PA model (single compilation — both PA nodes share executor cache)
    past_len_count = 0;
    prepare();
    init_all_kv_caches(1024 / 32);
    std::vector<ov::Tensor> actualOutputs;
    int idx = 0;
    for (auto&& shapes : targetStaticShapes) {
        generate(idx++, true, shapes, false, false);
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
        auto tensor = inferRequest.get_output_tensor(1);
        ov::Tensor copy{tensor.get_element_type(), tensor.get_shape()};
        tensor.copy_to(copy);
        actualOutputs.push_back(copy);
    }

    // Run SDPA reference for PA2 (head_size=hs2)
    past_len_count = 0;
    auto saved_function = function;
    function = functionRefs;
    prepare();
    std::vector<ov::Tensor> expectedOutputs;
    idx = 0;
    for (auto&& shapes : targetStaticShapes2_) {
        generate(idx++, false, shapes, false, false);
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
        auto tensor = inferRequest.get_output_tensor(0);
        ov::Tensor copy{tensor.get_element_type(), tensor.get_shape()};
        tensor.copy_to(copy);
        expectedOutputs.push_back(copy);
    }
    reset();
    function = saved_function;

    ASSERT_EQ(actualOutputs.size(), expectedOutputs.size());
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {
const std::vector<InputShapes> inputShapesCacheCollision = {{
    // [L, B=1, H=4, S=256] — PA1 head_size; PA2 uses S=512
    {{-1, 1, 4, 256}, {{10, 1, 4, 256}, {1, 1, 4, 256}}},
    {{-1, 1, 4, 256}, {{0, 1, 4, 256}, {10, 1, 4, 256}}},
}};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnExecutorCacheCollision,
                         PagedAttnCacheCollisionTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(inputShapesCacheCollision),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(ov::AnyMap{
                                                {ov::intel_cpu::enable_sage_attn.name(), false}}),
                                            ::testing::Values(false)),
                         PagedAttnTestBase::getTestCaseName);
}  // namespace

}  // namespace test
}  // namespace ov
