// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/common_utils.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "subgraphs_builders.hpp"
#include "shared_test_classes/base/utils/compare_results.hpp"

#include "openvino/runtime/intel_gpu/properties.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

namespace {
using ov::test::InputShape;

struct Params {
    bool is_caching_test;
    bool fuse_cache_reorder;
    bool build_state_initializer;
    bool with_mask;
    bool with_scale;
    bool causal;
    size_t batch;
    ov::element::Type model_element_type;
    size_t num_iter;
    size_t num_groups;
    bool set_state_on_each_iter;
    int32_t initial_batch;
    std::vector<int64_t> qkv_order;
};

class SDPAWithKVCacheTest: public ::testing::Test, public ::testing::WithParamInterface<Params> {
public:
    void test_smoke_multiple_iterations_stateful(const Params& p) {
    #if defined(ANDROID)
        GTEST_SKIP();
    #endif
        auto core = ov::test::utils::PluginCache::get().core();

        ov::AnyMap properties = {
            ov::hint::inference_precision(ov::element::f16),
            ov::intel_gpu::hint::enable_sdpa_optimization(true)
        };

        std::string cache_dir_name;
        if (p.is_caching_test) {
            std::stringstream ss;
            ss << "gpu_model_cache_" << std::hash<std::string>{}(
                  std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) +
                  std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
            std::string cache_dir_name = ss.str();
            ov::test::utils::removeFilesWithExt(cache_dir_name, "blob");
            ov::test::utils::removeFilesWithExt(cache_dir_name, "cl_cache");
            ov::test::utils::removeDir(cache_dir_name);
            properties.insert(ov::cache_dir(cache_dir_name));
        }

        const size_t n_heads = 32;
        const size_t n_features = 10;
        const size_t context_size = 20;

        const std::vector<int64_t>& qkv_order = p.qkv_order;

        ov::element::Type element_type = p.model_element_type;

        const bool stateful = true;
        const bool causal = false;
        const bool with_mask = false;
        const bool with_scale = false;

        auto model = tests::make_llm_kv_cache_sdpa_pattern(p.build_state_initializer ? ov::Dimension::dynamic() : p.batch,
                                                           n_heads,
                                                           n_features,
                                                           element_type,
                                                           qkv_order,
                                                           causal,
                                                           with_mask,
                                                           with_scale,
                                                           stateful,
                                                           p.fuse_cache_reorder,
                                                           p.build_state_initializer && stateful,
                                                           p.num_groups);
        auto ref_model = tests::make_llm_kv_cache_sdpa_pattern(p.build_state_initializer ? ov::Dimension::dynamic() : p.batch,
                                                               n_heads,
                                                               n_features,
                                                               element_type,
                                                               qkv_order,
                                                               causal,
                                                               with_mask,
                                                               with_scale,
                                                               !stateful,
                                                               p.fuse_cache_reorder,
                                                               p.build_state_initializer && !stateful,
                                                               p.num_groups);

        ov::pass::Manager manager;
        // Decompose ScaledDotProductAttention
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(ref_model);


        if (p.is_caching_test) {
            core->compile_model(model, ov::test::utils::DEVICE_GPU, properties);
        }
        auto compiled_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, properties);

        auto in_q = model->get_parameters().at(0);
        auto in_k_new_token = model->get_parameters().at(1);
        auto in_v_new_token = model->get_parameters().at(2);
        auto in_beam_idx = p.fuse_cache_reorder ? model->get_parameters().at(3) : nullptr;

        auto output = model->get_results().at(0);

        auto beam_idx_shape = ov::Shape{p.batch};

        auto adjust_qkv_shape = [&qkv_order](ov::Shape shape) {
            ov::Shape res(shape.size(), 0);
            for (size_t i = 0; i < shape.size(); i++) {
                res[qkv_order[i]] = shape[i];
            }

            return res;
        };

        auto get_ref_results = [&ref_model, &p](const ov::Tensor& k_cache,
                                                const ov::Tensor& v_cache,
                                                const ov::Tensor& k_new_token_data,
                                                const ov::Tensor& v_new_token_data,
                                                const ov::Tensor& q_data,
                                                const ov::Tensor& beam_idx_data,
                                                const ov::Shape& beam_idx_shape) {
            auto q = ref_model->get_parameters().at(0);
            auto k_past = ref_model->get_parameters().at(1);
            auto v_past = ref_model->get_parameters().at(2);
            auto k_new_token = ref_model->get_parameters().at(3);
            auto v_new_token = ref_model->get_parameters().at(4);
            auto beam_idx = p.fuse_cache_reorder ? ref_model->get_parameters().at(5)  : nullptr;
            std::map<ov::Output<ov::Node>, ov::PartialShape> input_shapes = {
                {q, q_data.get_shape()},
                {k_past, k_cache.get_shape()},
                {v_past, v_cache.get_shape()},
                {k_new_token, k_new_token_data.get_shape()},
                {v_new_token, v_new_token_data.get_shape()},
            };
            std::map<std::shared_ptr<ov::Node>, ov::Tensor> inputs = {
                {q, q_data},
                {k_past, k_cache},
                {v_past, v_cache},
                {k_new_token, k_new_token_data},
                {v_new_token, v_new_token_data},
            };
            if (p.fuse_cache_reorder) {
                input_shapes[beam_idx] = beam_idx_shape;
                inputs.emplace(beam_idx, beam_idx_data);
            }
            ref_model->reshape(input_shapes);
            return ov::test::utils::infer_on_template(ref_model, inputs);
        };

        ov::element::Type inference_precision = core->get_property(ov::test::utils::DEVICE_GPU, ov::hint::inference_precision);
        auto compare_tensors = [&model, &inference_precision](const std::vector<ov::Tensor> expected, const std::vector<ov::Tensor>& actual) {
            ASSERT_EQ(expected.size(), actual.size());
            auto compareMap = ov::test::utils::getCompareMap();
            for (size_t i = 0; i < expected.size(); i++) {
                auto it = compareMap.find(ov::op::v13::ScaledDotProductAttention::get_type_info_static());
                ASSERT_NE(it, compareMap.end());
                it->second(model->get_result(), i, inference_precision, expected[i], actual[i], 1e-1f, 1e-1f, 1.f, 1.f);
            }
        };

        auto infer_request = compiled_model.create_infer_request();

        auto sdpa_out = infer_request.get_tensor(output);
        auto tensor_q = infer_request.get_tensor(in_q);
        auto tensor_k_new_token = infer_request.get_tensor(in_k_new_token);
        auto tensor_v_new_token = infer_request.get_tensor(in_v_new_token);

        auto generator = ov::test::utils::InputGenerateData(-0.5, 1, 30000, 1);

        for (size_t num_repeats = 0; num_repeats < 2; num_repeats++) {
            ov::Tensor ref_k_cache;
            ov::Tensor ref_v_cache;
            size_t cache_size = 0;
            {
                // first infer
                size_t init_batch = p.initial_batch == -1 ? p.batch : static_cast<size_t>(p.initial_batch);
                const ov::Shape new_token_size_initial = adjust_qkv_shape({init_batch, n_heads / p.num_groups, context_size, n_features});
                const ov::Shape kv_cache_size_initial = adjust_qkv_shape({init_batch, n_heads / p.num_groups, cache_size, n_features});
                const ov::Shape q_in_size_initial = adjust_qkv_shape({init_batch, n_heads, context_size, n_features});

                auto k_new_token_data = ov::test::utils::create_and_fill_tensor(element_type, new_token_size_initial, generator);
                auto v_new_token_data = ov::test::utils::create_and_fill_tensor(element_type, new_token_size_initial, generator);
                auto q_data = ov::test::utils::create_and_fill_tensor(element_type, q_in_size_initial, generator);

                tensor_q.set_shape(q_data.get_shape());
                tensor_k_new_token.set_shape(k_new_token_data.get_shape());
                tensor_v_new_token.set_shape(v_new_token_data.get_shape());

                q_data.copy_to(tensor_q);
                k_new_token_data.copy_to(tensor_k_new_token);
                v_new_token_data.copy_to(tensor_v_new_token);

                auto init_beam_idx_shape = ov::Shape{init_batch};
                auto init_beam_idx_data_0 = ov::Tensor(ov::element::i32, init_beam_idx_shape);
                for (size_t i = 0; i < init_batch; i++) {
                    init_beam_idx_data_0.data<int32_t>()[i] = 0;
                }

                if (p.fuse_cache_reorder) {
                    infer_request.set_tensor(in_beam_idx, init_beam_idx_data_0);
                }

                ref_k_cache = ov::Tensor(element_type, kv_cache_size_initial);
                ref_v_cache = ov::Tensor(element_type, kv_cache_size_initial);

                auto ref_results = get_ref_results(ref_k_cache, ref_v_cache,
                                                   k_new_token_data, v_new_token_data,
                                                   q_data,
                                                   init_beam_idx_data_0, init_beam_idx_shape);
                ref_k_cache = ref_results[1];
                ref_v_cache = ref_results[2];

                infer_request.infer();

                compare_tensors({ ref_results[0] }, {sdpa_out});

                cache_size += context_size;
            }

            auto beam_idx_data_0 = ov::Tensor(ov::element::i32, beam_idx_shape);
            auto beam_idx_data_1 = ov::Tensor(ov::element::i32, beam_idx_shape);
            auto beam_idx_data_2 = ov::Tensor(ov::element::i32, beam_idx_shape);
            auto beam_idx_data_init = ov::Tensor(ov::element::i32, beam_idx_shape);
            for (size_t i = 0; i < p.batch; i++) {
                beam_idx_data_0.data<int32_t>()[i] = i;
                beam_idx_data_1.data<int32_t>()[i] = p.batch - i - 1;
                beam_idx_data_2.data<int32_t>()[i] = 0;
            }

            std::vector<ov::Tensor> beam_idx_data_array = {
                beam_idx_data_0,
                beam_idx_data_1,
                beam_idx_data_2,
            };

            const size_t input_tokens = 1;
            const ov::Shape new_token_size = adjust_qkv_shape({p.batch, n_heads / p.num_groups, input_tokens, n_features});
            size_t context_length = cache_size + input_tokens;
            for (size_t i = 0; i < p.num_iter; i++, context_length += input_tokens) {
                ov::Shape q_in_size_loop = adjust_qkv_shape({p.batch, n_heads, input_tokens, n_features});
                auto k_new_token_data = ov::test::utils::create_and_fill_tensor(element_type, new_token_size, generator);
                auto v_new_token_data = ov::test::utils::create_and_fill_tensor(element_type, new_token_size, generator);
                auto q_data = ov::test::utils::create_and_fill_tensor(element_type, q_in_size_loop, generator);
                size_t beam_idx_array_idx = i == 0 ? 2 : i % 2;
                if (p.fuse_cache_reorder) {
                    infer_request.set_tensor(in_beam_idx, beam_idx_data_array[beam_idx_array_idx]);
                }

                auto ref_results = get_ref_results(ref_k_cache, ref_v_cache,
                                                   k_new_token_data, v_new_token_data,
                                                   q_data,
                                                   beam_idx_data_array[beam_idx_array_idx], beam_idx_shape);

                ref_k_cache = ref_results[1];
                ref_v_cache = ref_results[2];

                tensor_q.set_shape(q_data.get_shape());
                tensor_k_new_token.set_shape(k_new_token_data.get_shape());
                tensor_v_new_token.set_shape(v_new_token_data.get_shape());

                q_data.copy_to(tensor_q);
                k_new_token_data.copy_to(tensor_k_new_token);
                v_new_token_data.copy_to(tensor_v_new_token);

                infer_request.infer();

                compare_tensors({ ref_results[0] }, { sdpa_out });
            }

            auto variables = infer_request.query_state();
            std::vector<ov::Tensor> states;
            for (auto& variable : variables) {
                auto state = variable.get_state();
                ASSERT_EQ(state.get_element_type(), element_type);
                states.push_back(state);
            }
            compare_tensors({ ref_k_cache, ref_v_cache }, states);
            infer_request.reset_state();
        }

        if (p.is_caching_test) {
            ov::test::utils::removeFilesWithExt(cache_dir_name, "blob");
            ov::test::utils::removeFilesWithExt(cache_dir_name, "cl_cache");
            ov::test::utils::removeDir(cache_dir_name);
        }
    }

    static std::string get_test_case_name(::testing::TestParamInfo<Params> obj) {
        Params p = obj.param;

        std::ostringstream result;
        result << "caching=" << p.is_caching_test << "_";
        result << "fuse_reorder=" << p.fuse_cache_reorder << "_";
        result << "state_init=" << p.build_state_initializer << "_";
        result << "batch=" << p.batch << "_";
        result << "et=" << p.model_element_type << "_";
        result << "num_iter=" << p.num_iter << "_";
        result << "num_groups=" << p.num_groups << "_";
        result << "initial_batch=" << p.initial_batch << "_";
        result << "reset_state=" << p.set_state_on_each_iter << "_";
        result << "qkv_order=" << ov::test::utils::vec2str(p.qkv_order);
        return result.str();
    }
};

TEST_P(SDPAWithKVCacheTest, MultipleIterationStateful) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    test_smoke_multiple_iterations_stateful(GetParam());
}

std::vector<Params> get_test_params() {
    std::vector<Params> p;
    const bool caching = true;
    const bool with_rearrange = true;
    const bool with_init = true;
    const bool state_check = true;
    const bool with_mask = true;
    const bool with_scale = true;
    const bool causal = true;

    p.push_back({!caching, with_rearrange, with_init, !with_mask, !with_scale, !causal, 1, ov::element::Type_t::f16, 10, 1, !state_check, 1, {0, 1, 2, 3}});
    // p.push_back({!caching, with_rearrange, with_init, !with_mask, !with_scale, !causal, 1, ov::element::Type_t::f16, 10, 1, !state_check, 1, {1, 2, 0, 3}});
    p.push_back({!caching, with_rearrange, with_init, !with_mask, !with_scale, !causal, 1, ov::element::Type_t::f16, 10, 1, !state_check, 1, {0, 2, 1, 3}});
    return p;
}

INSTANTIATE_TEST_SUITE_P(smoke, SDPAWithKVCacheTest, ::testing::ValuesIn(get_test_params()), SDPAWithKVCacheTest::get_test_case_name);

} // namespace
