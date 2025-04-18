#include "simple_llm_pipeline.hpp"

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

namespace {
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
} // anonymous namespace

void SimpleLLMPipeline::initialize(const std::string& model_path, ov::Core& core, const ov::AnyMap& config) {
    ov::AnyMap properties(config);
    m_max_prompt_len = get_or_default(properties, "NPUW_LLM_MAX_PROMPT_LEN", 1024);
    m_min_response_len = get_or_default(properties, "NPUW_LLM_MIN_RESPONSE_LEN", 4);
    m_kvcache_total = m_max_prompt_len + m_min_response_len;

    update_config(properties, {"NPU_USE_NPUW", "YES"});
    update_config(properties, {"NPUW_LLM", "YES"});

    const uint32_t m_batch_dim = get_or_default(properties, "NPUW_LLM_BATCH_DIM", 0);
    const uint32_t m_seq_len_dim = get_or_default(properties, "NPUW_LLM_SEQ_LEN_DIM", 2);

    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    m_compiled_model =
        std::make_shared<ov::CompiledModel>(core.compile_model(model, "NPU",
        properties));
    m_request = std::make_shared<ov::InferRequest>(m_compiled_model->create_infer_request());
}

std::vector<int64_t> SimpleLLMPipeline::generate(const std::vector<int64_t>& input_ids_vec) {
    std::vector<int64_t> result;

    // NB: Check if there is enough space in KV-cache to process input prompt
    auto prompt_len = input_ids_vec.size();
    if (prompt_len > m_max_prompt_len) {
        OPENVINO_THROW("Simple LLM pipeline may only process prompts up to "
                        + std::to_string(m_max_prompt_len) + " tokens. "
                        + "Set the \"MAX_PROMPT_LEN\" config option to increase the limit.");
    }

    std::vector<uint64_t> input_ids_vec_copy(input_ids_vec.begin(), input_ids_vec.end());
    ov::Tensor input_ids(ov::element::i64, ov::Shape{1, input_ids_vec.size()}, input_ids_vec_copy.data());

    auto shape = input_ids.get_shape();
    ov::Tensor attention_mask{input_ids.get_element_type(), shape};
    std::fill_n(attention_mask.data<int64_t>(), shape[0] * shape[1], 1);

    ov::Tensor position_ids{ov::element::i64, input_ids.get_shape()};
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + shape[0] * shape[1], 0);

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
    std::vector<int64_t> attention_mask_data(prompt_len, 1);
    m_request->set_tensor("input_ids", ov::Tensor(ov::element::i64, ov::Shape{1,1},  reinterpret_cast<void*>(&input_ids_data)));
    m_request->set_tensor("position_ids", ov::Tensor(ov::element::i64, ov::Shape{1,1}, reinterpret_cast<void*>(&position_ids_data)));

    std::size_t total_generated_tokens{1};
    // Assume that we don't want to produce more than min response len in simple pipeline.
    for (; total_generated_tokens < m_min_response_len; total_generated_tokens++) {
        // KV Cache is full, no further generation is possible
        if (position_ids_data + 1 == m_kvcache_total) {
            break;
        }

        // Just change the variables here, as pointers to them are already set to corresponding tensors
        input_ids_data = out_token;
        ++position_ids_data;
        // However, attention_mask changes its shape on each iteration, it should be re-set explicitly
        attention_mask_data.push_back(1);
        m_request->set_tensor("attention_mask", ov::Tensor(ov::element::i64, ov::Shape{1,attention_mask_data.size()}, (void*)&attention_mask_data[0]));

        m_request->infer();

        out_token = get_token_as_max_logit(m_request->get_tensor("logits"));
        result.push_back(out_token);
    }
    return result;
}
