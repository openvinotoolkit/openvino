// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_tests.hpp"
#include "comparators/nrmse.hpp"
#include "openvino/util/shared_object.hpp"
#include "openvino/util/file_util.hpp"

#include <filesystem>
#include <algorithm>

using namespace testing;
using namespace ov::npuw::tests;
using namespace ov::intel_npu::npuw;

#define PRINT_ARRAY(a, len)              \
    std::cout << #a << std::endl;        \
    for (int i = 0; i < len; i++) {      \
        std::cout << a[i] << ", ";       \
    }                                    \
    std::cout << std::endl;

template <typename T>
T get_or_default(ov::AnyMap& config, const std::string& key, const T& default_value) {
    if (auto it = config.find(key); it != config.end()) {
        return config[key].as<T>();
    }
    return default_value;
}

void update_config(ov::AnyMap& config, const std::pair<std::string, ov::Any>& pair) {
    if (config.count(pair.first) == 0) {
        config.insert(pair);
    }
}

ov::Tensor make_tensor_slice(ov::Tensor tensor, size_t dim, size_t start_pos, size_t end_pos) {
    ov::Shape start_shape(std::vector<size_t>(tensor.get_shape().size(), 0u));
    start_shape[dim] = start_pos;
    ov::Shape end_shape = tensor.get_shape();
    end_shape[dim] = end_pos;
    return ov::Tensor(tensor, start_shape, end_shape);
}

int64_t get_token_as_max_logit(ov::Tensor logits) {
    size_t vocab_size = logits.get_shape().back();
    size_t sequence_offset = (logits.get_shape()[1] - 1) * vocab_size;
    const float* logits_data = logits.data<const float>() + sequence_offset;
    int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
    return out_token;
}

class SimpleLLMPipeline {
public:
    void initialize(const std::string& model_path, ov::Core& core, const ov::AnyMap& config) {
        ov::AnyMap properties(config);
        const uint32_t max_prompt_len = get_or_default(properties, "NPUW_LLM_MAX_PROMPT_LEN", 1024);
        const uint32_t min_response_len = get_or_default(properties, "NPUW_LLM_MIN_RESPONSE_LEN", 16);
        m_max_prompt_len = max_prompt_len;
        m_kvcache_total = max_prompt_len + min_response_len;

        update_config(properties, {"NPU_USE_NPUW", "YES"});
        update_config(properties, {"NPUW_LLM", "YES"});

        const uint32_t m_batch_dim = get_or_default(properties, "NPUW_LLM_BATCH_DIM", 0);
        const uint32_t m_seq_len_dim = get_or_default(properties, "NPUW_LLM_SEQ_LEN_DIM", 2);

        // Replace CACHE_DIR option if NPUW is enabled
        if (properties.count("CACHE_DIR") != 0u) {
            std::string cache_dir = properties["CACHE_DIR"].as<std::string>();
            properties.emplace("NPUW_CACHE_DIR", cache_dir);
            properties.erase("CACHE_DIR");
        }

        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        m_compiled_model =
            std::make_shared<ov::CompiledModel>(core.compile_model(model, "NPU",
            properties));
        m_request = std::make_shared<ov::InferRequest>(m_compiled_model->create_infer_request());
    }

    std::vector<int64_t> generate(const std::vector<int64_t>& input_ids_vec) {
        std::vector<int64_t> result;

        PRINT_ARRAY(input_ids_vec, input_ids_vec.size());
        ov::Tensor input_ids(ov::element::i64, ov::Shape{1, input_ids_vec.size()},
                             reinterpret_cast<void*>(const_cast<int64_t*>((input_ids_vec.data()))));

        auto shape = input_ids.get_shape();
        ov::Tensor attention_mask{input_ids.get_element_type(), shape};
        std::fill_n(attention_mask.data<int64_t>(), shape[0] * shape[1], 1);
        PRINT_ARRAY(attention_mask.data<int64_t>(), shape[0] * shape[1]);

        // NB: Check if there is enough space in KV-cache to process input prompt
        auto prompt_len = input_ids.get_size();
        if (prompt_len > m_max_prompt_len) {
            OPENVINO_THROW("Simple LLM pipeline may only process prompts up to "
                        + std::to_string(m_max_prompt_len) + " tokens. "
                        + "Set the \"MAX_PROMPT_LEN\" config option to increase the limit.");
        }

        ov::Tensor position_ids{ov::element::i64, input_ids.get_shape()};
        // initialize_position_ids harcodes seq_len
        const size_t seq_length = attention_mask.get_shape()[1];

        const int64_t* attention_mask_raw_data = attention_mask.data<int64_t>();
        int64_t* position_ids_raw_data = position_ids.data<int64_t>();

        int64_t sum = 0;
        for (size_t i = 0; i < seq_length; i++) {
            const size_t element_offset = seq_length + i;
            position_ids_raw_data[i] = sum;
            if (attention_mask_raw_data[i] == 1) {
                sum += 1;
            }
        }
        PRINT_ARRAY(position_ids_raw_data, input_ids.get_shape()[0] * input_ids.get_shape()[1]);
        m_request->set_tensor("input_ids", input_ids);
        m_request->set_tensor("attention_mask", attention_mask);
        m_request->set_tensor("position_ids", position_ids);

        m_request->infer();

        auto padded_logits = m_request->get_tensor("logits");
        auto logits = padded_logits;
        auto padded_sequence_len = padded_logits.get_shape()[1];
        if (padded_sequence_len > 1) {
            // If SliceOut is not applied:
            logits = make_tensor_slice(padded_logits, 1, padded_sequence_len - prompt_len, padded_sequence_len);
        }

        // Get token with greedy search
        int64_t out_token = get_token_as_max_logit(logits);
        result.push_back(out_token);

        // Create variables for new input_ids and position_ids
        int64_t input_ids_data = -1;
        int64_t position_ids_data = prompt_len - 1;
        std::vector<int64_t> attention_mask_data(prompt_len - 1, 1);
        m_request->set_tensor("input_ids", ov::Tensor(ov::element::i64, ov::Shape{1,1},  reinterpret_cast<void*>(&input_ids_data)));
        m_request->set_tensor("position_ids", ov::Tensor(ov::element::i64, ov::Shape{1,1}, reinterpret_cast<void*>(&position_ids_data)));

        std::size_t total_generated_tokens{1};
        // parametrize
        for (; total_generated_tokens < 16; total_generated_tokens++) {
            // KV Cache is full, no further generation is possible
            if (position_ids_data + 1 == m_kvcache_total) {
                break;
            }

            // Just change the variables here, as pointers to them are already set to corresponding tensors
            input_ids_data = out_token;
            ++position_ids_data;
            // However, attention_mask changes its shape on each iteration, it should be re-set explicitly
            attention_mask_data.push_back(1);
            PRINT_ARRAY(attention_mask_data, attention_mask_data.size());
            m_request->set_tensor("attention_mask", ov::Tensor(ov::element::i64, ov::Shape{1,attention_mask_data.size()}, (void*)&attention_mask_data[0]));

            m_request->infer();

            out_token = get_token_as_max_logit(m_request->get_tensor("logits"));
            result.push_back(out_token);
        }
        return result;
    }
private:
    uint32_t m_batch_dim{0};
    uint32_t m_seq_len_dim{0};
    uint32_t m_max_prompt_len{0};
    uint32_t m_kvcache_total{0};
    std::shared_ptr<ov::CompiledModel> m_compiled_model;
    std::shared_ptr<ov::InferRequest> m_request;
};

using LLMTestParams = std::tuple<std::string, ov::AnyMap, std::vector<int64_t>>;
class E2EAccuracyTest : public ::testing::TestWithParam<LLMTestParams> {
public:
    ov::Core core;
    ov::AnyMap use_npuw_props;
    std::shared_ptr<ov::Model> model;

    void SetUp() override {
        // Register TEMPLATE plugin in OpenVINO:
        auto plugin_path =
            ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                            std::string(ov::test::utils::TEMPLATE_LIB)
                                            + OV_BUILD_POSTFIX);
        if (!ov::util::file_exists(plugin_path)) {
            OPENVINO_THROW("Plugin: " + plugin_path + " does not exists!");
        }
        core.register_plugin(plugin_path, ov::test::utils::DEVICE_TEMPLATE);

        auto param = GetParam();
        ov::AnyMap config;
        std::tie(m_model_path, config, m_reference) = param;
        config["NPUW_DEVICES"] = "TEMPLATE";
        m_simple_llm.initialize(m_model_path, core, config);
    }

    void generate() {
        // Input prompt: What number is bigger, 5.8 or 5.11?
// input_ids:                  [|3838 1372 374 11243 11 220 20 13 23 476 220 20 13 16 16 30 |]
// attention_mask:             [|1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 |]
// position_ids:               [|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 |]
// new_token :                 16515
        std::vector<int64_t> input_ids = {3838, 1372, 374, 11243, 11, 220, 20, 13,
                                          23, 476, 220, 20, 13, 16, 16, 30};
        m_actual = m_simple_llm.generate(input_ids);
        PRINT_ARRAY(m_actual, m_actual.size());
    }

    void accurate() {
        for (auto i = 0; i < m_actual.size(); ++i) {
            ASSERT_EQ(m_actual[i], m_reference[i]);
        }
    }

private:
    std::string m_model_path;
    SimpleLLMPipeline m_simple_llm;
    std::vector<int64_t> m_actual;
    std::vector<int64_t> m_reference;
};

TEST_P(E2EAccuracyTest, DefaultConfigIsAccurate) {
    generate();
    accurate();
}

INSTANTIATE_TEST_SUITE_P(AccuracyNPUW, E2EAccuracyTest,
    ::testing::Combine(testing::Values("C:\\apronina\\models\\TinyLlama\\openvino_model.xml"),
                       testing::Values(ov::AnyMap{}),
                       testing::Values(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})));