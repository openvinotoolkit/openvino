// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/subgraph_builders/llm_builders.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/utils/compare_results.hpp"
#include <optional>

namespace {
using ov::test::InputShape;

using KVCacheTestParams = std::tuple<std::vector<InputShape>,  // input shapes
                                     ov::element::Type>;       // in/out type

class KVCacheTest : public testing::WithParamInterface<KVCacheTestParams>,
                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string get_test_case_name(testing::TestParamInfo<KVCacheTestParams> obj) {
        const auto& [input_shapes, element_type] = obj.param;

        std::ostringstream result;
        for (const auto& shape : input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : input_shapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "precision=" << element_type;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [input_shapes, element_type] = GetParam();

        init_input_shapes(input_shapes);

        inType = outType = element_type;

        function = ov::test::utils::make_llm_kv_cache_pattern(inputDynamicShapes[0][0], inputDynamicShapes[0][1], inputDynamicShapes[0][3], element_type);
    }
};

TEST_P(KVCacheTest, Inference) {
    run();
}

TEST_P(KVCacheTest, Inference_cached) {
    std::stringstream ss;
    ss << "gpu_model_cache_" << std::hash<std::string>{}(
          std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) +
          std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::string cacheDirName = ss.str();
    {
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
        core->set_property(ov::cache_dir(cacheDirName));
        configuration.insert(ov::cache_dir(cacheDirName));
        compile_model();
    }
    {
        run();
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
    }
}

const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};

const std::vector<std::vector<InputShape>> input_shapes_basic = {
    {
        {{-1, 32, -1, 80}, { {1, 32, 0, 80}, {1, 32, 20, 80} }},
        {{-1, -1, 32, 80}, { {1, 20, 32, 80}, {1, 1, 32, 80} }},
        {{-1, 32, -1, -1}, { {1, 32, 1, 20}, {1, 32, 1, 21} }}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_GPU_Dynamic,
                         KVCacheTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(precisions)),
                         KVCacheTest::get_test_case_name);

class KVCacheTests: public ::testing::Test {
    public:
    void test_smoke_multipleIterations(bool is_caching_test) {
    #if defined(ANDROID)
        GTEST_SKIP();
    #endif
        auto core = ov::test::utils::PluginCache::get().core();
        ov::AnyMap properties = {
            ov::hint::inference_precision(ov::element::f16)
        };
        std::string cacheDirName;
        if (is_caching_test) {
            std::stringstream ss;
            ss << "gpu_model_cache_" << std::hash<std::string>{}(
                  std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) +
                  std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
            cacheDirName = ss.str();
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
            properties.insert(ov::cache_dir(cacheDirName));
        }

        const size_t batch = 1;
        const size_t n_heads = 32;
        const size_t n_features = 80;
        const size_t context_size = 20;
        size_t cache_size = 0;

        ov::element::Type element_type = ov::element::f16;

        auto model = ov::test::utils::make_llm_kv_cache_pattern(batch, n_heads, n_features, element_type);
        if (is_caching_test) {
            core->compile_model(model, ov::test::utils::DEVICE_GPU, properties);
        }
        auto compiled_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, properties);

        auto input0 = model->get_parameters().at(0);
        auto input1 = model->get_parameters().at(1);
        auto input2 = model->get_parameters().at(2);
        auto output0 = model->get_results().at(0);
        auto output1 = model->get_results().at(1);

        auto get_ref_results = [&](const ov::Tensor& kv_cache, const ov::Tensor& new_token_data,
                                                                   const ov::Tensor& matmul_data) {
            auto ref_model = model->clone();
            ov::Tensor kv_cache_copy(kv_cache.get_element_type(), kv_cache.get_shape());
            kv_cache.copy_to(kv_cache_copy);
            std::map<ov::Output<ov::Node>, ov::PartialShape> shapes = {
                {ref_model->input(0), kv_cache_copy.get_shape()},
                {ref_model->input(1), new_token_data.get_shape()},
                {ref_model->input(2), matmul_data.get_shape()}
            };
            ref_model->reshape(shapes);

            auto compiled_model_ref = core->compile_model(ref_model, ov::test::utils::DEVICE_TEMPLATE);
            auto inf_req_ref = compiled_model_ref.create_infer_request();
            inf_req_ref.set_tensor(input0, kv_cache_copy);
            inf_req_ref.set_tensor(input1, new_token_data);
            inf_req_ref.set_tensor(input2, matmul_data);
            inf_req_ref.infer();
            std::vector<ov::Tensor> results_ref;
            for (auto&& output : ref_model->get_results()) {
                results_ref.push_back(inf_req_ref.get_tensor(output));
            }
            return results_ref;
        };

        ov::element::Type inference_precision = core->get_property(ov::test::utils::DEVICE_TEMPLATE, ov::hint::inference_precision);
        auto compare_tensors = [&model, &inference_precision](const std::vector<ov::Tensor> expected, const std::vector<ov::Tensor>& actual) {
                ASSERT_EQ(expected.size(), actual.size());
                ASSERT_EQ(expected.size(), model->get_results().size());
                const auto& compareMap = ov::test::utils::getCompareMap();
                const auto& results = model->get_results();
                for (size_t j = 0; j < results.size(); j++) {
                    const auto result = results[j];
                    for (size_t i = 0; i < result->get_input_size(); ++i) {
                        std::shared_ptr<ov::Node> inputNode = result->get_input_node_shared_ptr(i);
                        auto it = compareMap.find(inputNode->get_type_info());
                        ASSERT_NE(it, compareMap.end());
                        it->second(inputNode, i, inference_precision, expected[j], actual[j], 1e-4f, 1e-4f, 1.f, 1.f);
                    }
                }
        };

        auto infer_request = compiled_model.create_infer_request();
        auto kv_cache_input = infer_request.get_tensor(output0);
        auto matmul_out = infer_request.get_tensor(output1);
        auto new_token_input = infer_request.get_tensor(input1);
        auto matmul_input = infer_request.get_tensor(input2);

        infer_request.set_tensor(input0, kv_cache_input);
        infer_request.set_tensor(input1, new_token_input);
        infer_request.set_tensor(input2, matmul_input);

        {
            const ov::Shape new_token_size_initial = {batch, context_size, n_heads, n_features};
            const ov::Shape kv_cache_size_initial = {batch, n_heads, cache_size, n_features};
            const ov::Shape matmul_in_size_initial = {batch, n_heads, context_size, context_size};

            auto new_token_data = ov::test::utils::create_and_fill_tensor(element_type, new_token_size_initial);
            auto matmul_data = ov::test::utils::create_and_fill_tensor(element_type, matmul_in_size_initial);

            kv_cache_input.set_shape(kv_cache_size_initial);
            new_token_input.set_shape(new_token_data.get_shape());
            matmul_input.set_shape(matmul_data.get_shape());

            new_token_data.copy_to(new_token_input);
            matmul_data.copy_to(matmul_input);

            auto ref_results = get_ref_results(kv_cache_input, new_token_data, matmul_data);

            infer_request.infer();

            compare_tensors(ref_results, {kv_cache_input, matmul_out});

            cache_size += context_size;
        }

        const size_t input_tokens = 1;
        const size_t niters = 10;
        const ov::Shape new_token_size = {batch, input_tokens, n_heads, n_features};
        size_t context_length = cache_size + input_tokens;
        for (size_t i = 0; i < niters; i++, context_length += input_tokens) {
            ov::Shape matmul_in_size_loop = {batch, n_heads, input_tokens, context_length};
            auto new_token_data = ov::test::utils::create_and_fill_tensor(element_type, new_token_size);
            auto matmul_data = ov::test::utils::create_and_fill_tensor(element_type, matmul_in_size_loop);
            auto ref_results = get_ref_results(kv_cache_input, new_token_data, matmul_data);

            new_token_input.set_shape(new_token_data.get_shape());
            matmul_input.set_shape(matmul_data.get_shape());
            new_token_data.copy_to(new_token_input);
            matmul_data.copy_to(matmul_input);

            infer_request.infer();

            compare_tensors(ref_results, {kv_cache_input, matmul_out});
        }

        if (is_caching_test) {
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
        }
    }

    /**
     * @brief Additional LLM KV cache trimming parameter
     */
    struct kv_cache_trim_params {
        /// \param Length of sequence that starts trimming
        int32_t trigger_len = 0;
        /// \param Length of sequence after trimmed
        int32_t trim_seq = 0;
    };

    /**
     * @brief Additional LLM KV cache reordering parameter
     */
    struct kv_cache_reorder_params {
        /// \param Source indices for reorder
        std::vector<int32_t> src_idx;
        /// \param Destination indices for reorder
        std::vector<int32_t> dst_idx;
    };

    void test_smoke_multipleIterations_stateful(bool is_caching_test,
                                                bool fuse_cache_reorder,
                                                bool build_state_initializer,
                                                size_t batch = 1,
                                                int64_t concat_axis = 2,
                                                ov::element::Type model_element_type = ov::element::f16,
                                                size_t num_iter = 10,
                                                size_t num_groups = 1,
                                                bool set_state_on_each_iter = false,
                                                int32_t initial_batch = -1,
                                                const kv_cache_trim_params* trim_params = nullptr,
                                                const kv_cache_reorder_params* reorder_params = nullptr) {
    #if defined(ANDROID)
        GTEST_SKIP();
    #endif
        auto core = ov::test::utils::PluginCache::get().core();

        ov::AnyMap properties = {
            ov::hint::inference_precision(ov::element::f16)
        };

        std::string cacheDirName;
        if (is_caching_test) {
            std::stringstream ss;
            ss << "gpu_model_cache_" << std::hash<std::string>{}(
                  std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) +
                  std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
            std::string cacheDirName = ss.str();
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
            properties.insert(ov::cache_dir(cacheDirName));
        }

        const size_t n_heads = 32;
        const size_t n_features = 10;
        const size_t context_size = 20;

        ov::element::Type element_type = model_element_type;

        const bool stateful = true;
        if (trim_params) {
            OPENVINO_ASSERT(trim_params->trim_seq > 0 && trim_params->trigger_len >= trim_params->trim_seq);
        }
        if (reorder_params) {
            OPENVINO_ASSERT(trim_params);
            OPENVINO_ASSERT(reorder_params->src_idx.size() == reorder_params->dst_idx.size());
            // make sure src/dst idx within range
            const auto src_idx_fit = std::all_of(reorder_params->src_idx.begin(), reorder_params->src_idx.end(), [&](const auto& idx) {
                return idx >= 0 && idx < trim_params->trigger_len;
            });
            OPENVINO_ASSERT(src_idx_fit);
            const auto dst_idx_fit = std::all_of(reorder_params->dst_idx.begin(), reorder_params->dst_idx.end(), [&](const auto& idx) {
                return idx >= 0 && idx < trim_params->trim_seq;
            });
            OPENVINO_ASSERT(dst_idx_fit);
        }

        auto model = ov::test::utils::make_llm_kv_cache_pattern(build_state_initializer ? ov::Dimension::dynamic() : batch,
                                                                n_heads,
                                                                n_features,
                                                                element_type,
                                                                concat_axis,
                                                                stateful,
                                                                fuse_cache_reorder,
                                                                build_state_initializer && stateful,
                                                                num_groups,
                                                                trim_params,
                                                                reorder_params);
        auto ref_model = ov::test::utils::make_llm_kv_cache_pattern(build_state_initializer ? ov::Dimension::dynamic() : batch,
                                                                    n_heads,
                                                                    n_features,
                                                                    element_type,
                                                                    concat_axis,
                                                                    !stateful,
                                                                    fuse_cache_reorder,
                                                                    build_state_initializer && !stateful,
                                                                    num_groups,
                                                                    trim_params,
                                                                    reorder_params);
        if (is_caching_test) {
            core->compile_model(model, ov::test::utils::DEVICE_GPU, properties);
        }
        ov::Shape unit_shape = {1};

        struct kv_cache_trim_state {
            const kv_cache_trim_params& trim;
            ov::Tensor seq_len;
            ov::Shape seq_len_shape;
            kv_cache_trim_state(const kv_cache_trim_params& trim_params) : trim(trim_params), seq_len_shape{1} {}
            virtual ~kv_cache_trim_state() {}
            virtual std::optional<size_t> update(const size_t past_seq_len) {
                if (past_seq_len >= static_cast<uint32_t>(trim.trigger_len)) {
                    seq_len.data<int32_t>()[0] = trim.trim_seq;
                    return trim.trim_seq;
                } else {
                    OPENVINO_ASSERT(past_seq_len < std::numeric_limits<int32_t>::max());
                    seq_len.data<int32_t>()[0] = static_cast<int32_t>(past_seq_len);
                    return std::nullopt;
                }
            }
        };
        struct kv_cache_reorder_state : kv_cache_trim_state {
            const kv_cache_reorder_params& reorder;
            ov::Tensor src_idx;
            ov::Tensor dst_idx;
            ov::Tensor src_idx_data;
            ov::Tensor dst_idx_data;
            ov::Shape src_shape;
            ov::Shape dst_shape;
            kv_cache_reorder_state(const kv_cache_trim_params& trim_params,
                                   const kv_cache_reorder_params& reorder_params,
                                   size_t batch,
                                   size_t n_heads,
                                   size_t n_features)
                : kv_cache_trim_state(trim_params), 
                  reorder(reorder_params),
                  src_shape{reorder.src_idx.size()},
                  dst_shape{batch, n_heads, reorder.dst_idx.size(), n_features} {
                src_idx_data = ov::Tensor(ov::element::i32, src_shape, reorder.src_idx.data());
                dst_idx_data = ov::Tensor(ov::element::i32, dst_shape);
                auto dst_ptr = dst_idx_data.data<int32_t>();
                for (size_t b = 0; b < batch; ++b) {
                    for (size_t h = 0; h < n_heads; ++h) {
                        for (const auto slice : reorder.dst_idx) {
                            dst_ptr = std::fill_n(dst_ptr, n_features, slice);
                        }
                    }
                }
            }
            ~kv_cache_reorder_state() override {}
            std::optional<size_t> update(const size_t past_seq_len) override {
                const auto new_len = kv_cache_trim_state::update(past_seq_len);
                if (new_len.has_value()) {
                    src_shape = {reorder.src_idx.size()};
                    dst_shape[2] = reorder.dst_idx.size();
                    src_idx.set_shape(src_shape);
                    dst_idx.set_shape(dst_shape);
                    src_idx_data.copy_to(src_idx);
                    dst_idx_data.copy_to(dst_idx);
                } else {
                    src_shape = {0};
                    dst_shape[2] = 0;
                    src_idx.set_shape(src_shape);
                    dst_idx.set_shape(dst_shape);
                }
                return new_len;
            }
        };
        std::unique_ptr<kv_cache_trim_state> extra_state;
        if (reorder_params) {
            extra_state = std::make_unique<kv_cache_reorder_state>(*trim_params, *reorder_params, batch, n_heads, n_features);
        } else if (trim_params) {
            extra_state = std::make_unique<kv_cache_trim_state>(*trim_params);
        }

        auto compiled_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, properties);

        size_t param_idx = 0;
        auto input0 = model->get_parameters().at(param_idx++);
        auto input1 = model->get_parameters().at(param_idx++);
        auto input_beam_idx = fuse_cache_reorder ? model->get_parameters().at(param_idx++) : nullptr;
        auto input_src_idx = reorder_params ? model->get_parameters().at(param_idx++) : nullptr;
        auto input_dst_idx = reorder_params ? model->get_parameters().at(param_idx++) : nullptr;
        auto input_trim = trim_params ? model->get_parameters().at(param_idx++) : nullptr;
        auto output0 = model->get_results().at(0);

        auto beam_idx_shape = ov::Shape{batch};

        auto get_ref_results = [&ref_model, fuse_cache_reorder, trim_params, reorder_params](const ov::Tensor& kv_cache,
                                                                                             const ov::Tensor& new_token_data,
                                                                                             const ov::Tensor& matmul_data,
                                                                                             const ov::Tensor& beam_idx_data,
                                                                                             const ov::Shape& beam_idx_shape,
                                                                                             const kv_cache_trim_state* extra_state) {
            size_t param_idx = 0;
            auto input0 = ref_model->get_parameters().at(param_idx++);
            auto input1 = ref_model->get_parameters().at(param_idx++);
            auto input2 = ref_model->get_parameters().at(param_idx++);
            auto input_beam_idx = fuse_cache_reorder ? ref_model->get_parameters().at(param_idx++) : nullptr;
            auto input_src_idx = reorder_params ? ref_model->get_parameters().at(param_idx++) : nullptr;
            auto input_dst_idx = reorder_params ? ref_model->get_parameters().at(param_idx++) : nullptr;
            auto input_trim = trim_params ? ref_model->get_parameters().at(param_idx++) : nullptr;
            std::map<ov::Output<ov::Node>, ov::PartialShape> input_shapes = {
                {input0, kv_cache.get_shape()},
                {input1, new_token_data.get_shape()},
                {input2, matmul_data.get_shape()}
            };
            std::map<std::shared_ptr<ov::Node>, ov::Tensor> inputs = {
                {input0, kv_cache},
                {input1, new_token_data},
                {input2, matmul_data}
            };
            if (fuse_cache_reorder) {
                input_shapes[input_beam_idx] = beam_idx_shape;
                inputs.emplace(input_beam_idx, beam_idx_data);
            }
            if (reorder_params) {
                const auto& extra = static_cast<const kv_cache_reorder_state&>(*extra_state);
                input_shapes[input_src_idx] = extra.src_shape;
                inputs.emplace(input_src_idx, extra.src_idx);
                input_shapes[input_dst_idx] = extra.dst_shape;
                inputs.emplace(input_dst_idx, extra.dst_idx);
            }
            if (trim_params) {
                const auto& extra = *extra_state;
                input_shapes[input_trim] = extra.seq_len_shape;
                inputs.emplace(input_trim, extra.seq_len);
            }
            ref_model->reshape(input_shapes);
            return ov::test::utils::infer_on_template(ref_model, inputs);
        };

        ov::element::Type inference_precision = core->get_property(ov::test::utils::DEVICE_GPU, ov::hint::inference_precision);
        auto compare_tensors = [&model, &inference_precision](const std::vector<ov::Tensor> expected, const std::vector<ov::Tensor>& actual) {
            ASSERT_EQ(expected.size(), actual.size());
            ASSERT_EQ(expected.size(), model->get_results().size());
            const auto& compareMap = ov::test::utils::getCompareMap();
            const auto& results = model->get_results();
            for (size_t j = 0; j < results.size(); j++) {
                const auto result = results[j];
                for (size_t i = 0; i < result->get_input_size(); ++i) {
                    std::shared_ptr<ov::Node> inputNode = result->get_input_node_shared_ptr(i);
                    auto it = compareMap.find(inputNode->get_type_info());
                    ASSERT_NE(it, compareMap.end());
                    it->second(inputNode, i, inference_precision, expected[j], actual[j], 1e-4f, 1e-4f, 1.f, 1.f);
                }
            }
        };

        auto infer_request = compiled_model.create_infer_request();
        auto matmul_out = infer_request.get_tensor(output0);
        auto new_token_input = infer_request.get_tensor(input0);
        auto matmul_input = infer_request.get_tensor(input1);

        if (extra_state) {
            extra_state->seq_len = infer_request.get_tensor(input_trim);
            infer_request.set_tensor(input_trim, extra_state->seq_len);
        }
        if (reorder_params) {
            auto& extra = static_cast<kv_cache_reorder_state&>(*extra_state);
            extra.src_idx = infer_request.get_tensor(input_src_idx);
            extra.dst_idx = infer_request.get_tensor(input_dst_idx);
            infer_request.set_tensor(input_src_idx, extra.src_idx);
            infer_request.set_tensor(input_dst_idx, extra.dst_idx);
        }

        infer_request.set_tensor(input0, new_token_input);
        infer_request.set_tensor(input1, matmul_input);

        for (size_t num_repeats = 0; num_repeats < 2; num_repeats++) {
            ov::Tensor ref_kv_cache;
            size_t cache_size = 0;
            {
                // first infer
                size_t init_batch = initial_batch == -1 ? batch : static_cast<size_t>(initial_batch);
                const ov::Shape new_token_size_initial = {init_batch, context_size, n_heads / num_groups, n_features};
                const ov::Shape kv_cache_size_initial = {init_batch, n_heads / num_groups, cache_size, n_features};
                const ov::Shape matmul_in_size_initial = {init_batch, n_heads, context_size, context_size};

                auto new_token_data = ov::test::utils::create_and_fill_tensor(element_type, new_token_size_initial);
                auto matmul_data = ov::test::utils::create_and_fill_tensor(element_type, matmul_in_size_initial);

                new_token_input.set_shape(new_token_data.get_shape());
                matmul_input.set_shape(matmul_data.get_shape());

                new_token_data.copy_to(new_token_input);
                matmul_data.copy_to(matmul_input);

                auto init_beam_idx_shape = ov::Shape{init_batch};
                auto init_beam_idx_data_0 = ov::Tensor(ov::element::i32, init_beam_idx_shape);
                for (size_t i = 0; i < init_batch; i++) {
                    init_beam_idx_data_0.data<int32_t>()[i] = 0;
                }

                if (fuse_cache_reorder) {
                    infer_request.set_tensor(input_beam_idx, init_beam_idx_data_0);
                }

                if (extra_state) {
                    cache_size = extra_state->update(cache_size).value_or(cache_size);
                }

                ref_kv_cache = ov::Tensor(element_type, kv_cache_size_initial);

                auto ref_results = get_ref_results(ref_kv_cache, new_token_data, matmul_data, init_beam_idx_data_0, init_beam_idx_shape, extra_state.get());
                ref_kv_cache = ref_results[0];

                infer_request.infer();

                compare_tensors({ ref_results[1] }, {matmul_out});

                cache_size += context_size;
            }

            auto beam_idx_data_0 = ov::Tensor(ov::element::i32, beam_idx_shape);
            auto beam_idx_data_1 = ov::Tensor(ov::element::i32, beam_idx_shape);
            auto beam_idx_data_2 = ov::Tensor(ov::element::i32, beam_idx_shape);
            auto beam_idx_data_init = ov::Tensor(ov::element::i32, beam_idx_shape);
            for (size_t i = 0; i < batch; i++) {
                beam_idx_data_0.data<int32_t>()[i] = i;
                beam_idx_data_1.data<int32_t>()[i] = batch - i - 1;
                beam_idx_data_2.data<int32_t>()[i] = 0;
            }

            std::vector<ov::Tensor> beam_idx_data_array = {
                beam_idx_data_0,
                beam_idx_data_1,
                beam_idx_data_2,
            };

            const size_t input_tokens = 1;
            const ov::Shape new_token_size = {batch, input_tokens, n_heads / num_groups, n_features};
            for (size_t i = 0; i < num_iter; i++) {
                if (extra_state) {
                    cache_size = extra_state->update(cache_size).value_or(cache_size);
                }
                size_t context_length = cache_size + input_tokens;
                ov::Shape matmul_in_size_loop = {batch, n_heads, input_tokens, context_length};
                auto new_token_data = ov::test::utils::create_and_fill_tensor(element_type, new_token_size);
                auto matmul_data = ov::test::utils::create_and_fill_tensor(element_type, matmul_in_size_loop);
                size_t beam_idx_array_idx = i == 0 ? 2 : i % 2;
                if (fuse_cache_reorder) {
                    infer_request.set_tensor(input_beam_idx, beam_idx_data_array[beam_idx_array_idx]);
                }
                auto ref_results =
                    get_ref_results(ref_kv_cache, new_token_data, matmul_data, beam_idx_data_array[beam_idx_array_idx], beam_idx_shape, extra_state.get());
                ref_kv_cache = ref_results[0];

                new_token_input.set_shape(new_token_data.get_shape());
                matmul_input.set_shape(matmul_data.get_shape());
                new_token_data.copy_to(new_token_input);
                matmul_data.copy_to(matmul_input);

                infer_request.infer();

                compare_tensors({ ref_results[1] }, {matmul_out});

                if (set_state_on_each_iter) {
                    auto state = infer_request.query_state()[0].get_state();
                    compare_tensors({ ref_kv_cache }, {state});
                    infer_request.query_state()[0].set_state(state);
                    auto state_1 = infer_request.query_state()[0].get_state();
                    compare_tensors({ ref_kv_cache }, {state_1});
                }

                cache_size = context_length;
            }

            auto state = infer_request.query_state()[0].get_state();
            ASSERT_EQ(state.get_element_type(), element_type);
            compare_tensors({ ref_kv_cache }, {state});

            infer_request.reset_state();
        }

        if (is_caching_test) {
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
        }
    }
};

TEST_F(KVCacheTests, smoke_multipleIterations) {
    this->test_smoke_multipleIterations(false);
}

TEST_F(KVCacheTests, smoke_multipleIterations_cached) {
    this->test_smoke_multipleIterations(true);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_no_gather_no_initializer_concat_neg_axis) {
    this->test_smoke_multipleIterations_stateful(false, false, false, 1, -2);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_no_gather_no_initializer_cached) {
    this->test_smoke_multipleIterations_stateful(true, false, false);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_gather_with_initializer) {
    this->test_smoke_multipleIterations_stateful(false, true, true);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_gather_with_initializer_cached) {
    this->test_smoke_multipleIterations_stateful(true, true, true);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_gather_with_initializer_gqa) {
    this->test_smoke_multipleIterations_stateful(false, true, true, 1, 2, ov::element::f16, 10, 4);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_gather_with_initializer_f32) {
    this->test_smoke_multipleIterations_stateful(false, true, true, 1, 2, ov::element::f32);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_gather_with_initializer_batch_3) {
    this->test_smoke_multipleIterations_stateful(false, true, true, 3);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_gather_with_initializer_batch_1_3) {
    this->test_smoke_multipleIterations_stateful(false, true, true, 3, 2, ov::element::f16, 10, 1, false, 1);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_gather_with_initializer_batch_1_5) {
    this->test_smoke_multipleIterations_stateful(false, true, true, 5, 2, ov::element::f16, 10, 1, false, 1);
}


TEST_F(KVCacheTests, smoke_multipleIterations_stateful_same_shape_after_reset) {
    this->test_smoke_multipleIterations_stateful(false, false, false, 1, 2, ov::element::f16, 0);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_with_set_state) {
    this->test_smoke_multipleIterations_stateful(false, true, true, 1, 2, ov::element::f16, 5, 1, true);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_trim) {
    kv_cache_trim_params trim;
    trim.trigger_len = 17;
    trim.trim_seq = 14;
    this->test_smoke_multipleIterations_stateful(false, false, true, 1, 2, ov::element::f16, 5, 1, true, 1, &trim);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_beam_trim) {
    kv_cache_trim_params trim;
    trim.trigger_len = 200;
    trim.trim_seq = 200;
    this->test_smoke_multipleIterations_stateful(false, true, true, 1, 2, ov::element::f16, 5, 1, true, 1, &trim);
}

TEST_F(KVCacheTests, smoke_multipleIterations_stateful_trim_reorder) {
    kv_cache_trim_params trim;
    kv_cache_reorder_params reorder;
    trim.trigger_len = 18;
    trim.trim_seq = 14;
    reorder.src_idx = {12, 13, 14};
    reorder.dst_idx = {10, 11, 12};
    this->test_smoke_multipleIterations_stateful(false, false, true, 1, 2, ov::element::f16, 5, 1, true, 1, &trim, &reorder);
}

class KVCacheIssueTests: public ::testing::Test {
public:
    void test_smoke_conflicted_memory_for_two_inf_req() {
    #if defined(ANDROID)
        GTEST_SKIP();
    #endif
        auto core = ov::test::utils::PluginCache::get().core();

        ov::AnyMap properties = {
            ov::hint::kv_cache_precision(ov::element::f16)
        };

        const size_t n_batch = 1;
        const size_t n_heads = 32;
        const size_t n_features = 10;
        const size_t context_size = 20;
        ov::element::Type element_type = ov::element::f16;

        const bool stateful = true;

        auto model = ov::test::utils::make_llm_kv_cache_pattern(n_batch, n_heads, n_features, element_type, 2, stateful, false, stateful);
        auto compiled_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, properties);

        auto input0 = model->get_parameters().at(0);
        auto input1 = model->get_parameters().at(1);

        auto ireq1 = compiled_model.create_infer_request();
        auto ireq2 = compiled_model.create_infer_request();

        auto ireq1_input0 = ov::test::utils::create_and_fill_tensor_real_distribution(element_type,
                                {n_batch, context_size, n_heads, n_features}, -0.5f, 0.5f, 1);
        auto ireq1_input1 = ov::test::utils::create_and_fill_tensor_real_distribution(element_type,
                                {n_batch, n_heads, context_size, context_size}, -0.5f, 0.5f, 1);
        // Create read-only tensor view to test inference with this const input data
        ireq1.set_tensor(input0, {ireq1_input0.get_element_type(), ireq1_input0.get_shape(), std::as_const(ireq1_input0).data()});
        ireq1.set_tensor(input1, ireq1_input1);

        auto ireq2_input0 = ov::test::utils::create_and_fill_tensor_real_distribution(element_type,
                                {n_batch, context_size + 1, n_heads, n_features}, -0.5f, 0.5f, 555);
        auto ireq2_input1 = ov::test::utils::create_and_fill_tensor_real_distribution(element_type,
                                {n_batch, n_heads, context_size + 1, context_size + 1}, -0.5f, 0.5f, 555);
        ireq2.set_tensor(input0, ireq2_input0);
        ireq2.set_tensor(input1, ireq2_input1);

        std::stringstream oss1;
        std::stringstream oss2;
        for (auto&& state : ireq1.query_state()) {
            state.reset();
        }
        ireq1.infer();
        for (auto&& state : ireq1.query_state()) {
            oss1.write(reinterpret_cast<char*>(state.get_state().data()), state.get_state().get_byte_size());
        }

        for (auto&& state : ireq2.query_state()) {
            state.reset();
        }
        ireq2.infer();
        for (auto&& state : ireq1.query_state()) {
            oss2.write(reinterpret_cast<char*>(state.get_state().data()), state.get_state().get_byte_size());
        }

        ASSERT_TRUE(oss1.str() == oss2.str());
    }
};

TEST_F(KVCacheIssueTests, conflicted_memory_for_two_inf_req) {
    this->test_smoke_conflicted_memory_for_two_inf_req();
}


} // namespace
