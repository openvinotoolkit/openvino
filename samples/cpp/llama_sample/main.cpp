// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <memory>
#include <sstream>
#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"
#include "llama_cpp/llama.h"
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;
inline double get_duration_ms_till_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
};

void llama_sample_softmax(llama_token_data_array * candidates) {
    // Sort the logits in descending order
    if (!candidates->sorted) {
        std::sort(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        });
        candidates->sorted = true;
    }

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }
}

void llama_top_k(llama_token_data_array* candidates, int k, size_t min_keep) {
    k = std::max(k, (int)min_keep);
    k = std::min(k, (int)candidates->size);

    // Sort scores in descending order
    if (!candidates->sorted) {
        auto comp = [](const llama_token_data& a, const llama_token_data& b) {
            return a.logit > b.logit;
        };
        if (k == (int)candidates->size) {
            std::sort(candidates->data, candidates->data + candidates->size, comp);
        } else {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        }
        candidates->sorted = true;
    }
    candidates->size = k;
}

llama_token llama_token_greedy(llama_token_data_array * candidates) {
    auto * max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit < b.logit;
    });

    llama_token result = max_iter->id;
    return result;
}

class tokenizer_from_llama_cpp
{
public:
    llama_model *tokenizer_model;
    llama_context *ctx;
    tokenizer_from_llama_cpp(std::string vocab_path);
    ~tokenizer_from_llama_cpp();
    std::vector<int> tokenizer(std::string prompt_text);
    std::string tokenizer_decode(int id);
    int get_llama_token_eos();
    int get_llama_n_vocab();
};

tokenizer_from_llama_cpp::tokenizer_from_llama_cpp(std::string vocab_path) {
    llama_backend_init(false);
    auto lparams = llama_context_default_params();
    lparams.vocab_only = true;
    tokenizer_model = llama_load_model_from_file(vocab_path.c_str(), lparams);
    if (tokenizer_model == NULL) {
        OPENVINO_THROW("Failed to load vocab ", vocab_path.c_str());
    }
    ctx = llama_new_context_with_model(tokenizer_model, lparams);
    if (ctx == NULL) {
        llama_free_model(tokenizer_model);
        OPENVINO_THROW("Failed to load vocab ", vocab_path.c_str());
    }
    const int n_vocab = llama_n_vocab(ctx);
    if (n_vocab != 32000) {
        llama_free_model(tokenizer_model);
        llama_free(ctx);
        OPENVINO_THROW("Expected 32000 tokens, got ", n_vocab);
    }
}

tokenizer_from_llama_cpp::~tokenizer_from_llama_cpp() {
    llama_free_model(tokenizer_model);
    llama_free(ctx);
}

std::vector<int> tokenizer_from_llama_cpp::tokenizer(std::string prompt_text) {
    std::vector<llama_token> prompt(prompt_text.size());
    const int n = llama_tokenize(ctx, prompt_text.c_str(), prompt.data(), int(prompt.size()), true);
    prompt.resize(n);

    return prompt;
}

std::string tokenizer_from_llama_cpp::tokenizer_decode(int id) {
    return llama_token_to_str(ctx , id);
}

int tokenizer_from_llama_cpp::get_llama_token_eos() {
    return llama_token_eos();
}

int tokenizer_from_llama_cpp::get_llama_n_vocab() {
    return llama_n_vocab(ctx);
}

/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 5) {
            slog::info << "Usage : " << argv[0] << " <path_to_model> <path_to_vocab_file> <prompt_text> <max_sequence_length>" << slog::endl;
            return EXIT_FAILURE;
        }

        const std::string args = TSTRING2STRING(argv[0]);
        const std::string model_path = TSTRING2STRING(argv[1]);
        const std::string vocab_file = TSTRING2STRING(argv[2]);
        const std::string prompt_str = TSTRING2STRING(argv[3]);
        const int max_sequence_length = strtol(argv[4], NULL, 10);
        auto startTime = Time::now();
        tokenizer_from_llama_cpp llama_tokenizer(vocab_file);
        std::vector<int> prompt = llama_tokenizer.tokenizer(prompt_str);
        auto duration_ms = get_duration_ms_till_now(startTime);
        slog::info << "Tokenizer took " << double_to_string(duration_ms) << " ms" << slog::endl;
        ov::Core core;
        slog::info << "Loading model files: " << model_path << slog::endl;
        startTime = Time::now();
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        duration_ms = get_duration_ms_till_now(startTime);
        slog::info << "Read model took " << double_to_string(duration_ms) << " ms" << slog::endl;
        startTime = Time::now();
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU", {{"PERFORMANCE_HINT", "LATENCY"}, {"NUM_STREAMS", 1}, {"CACHE_DIR", ""}});
        duration_ms = get_duration_ms_till_now(startTime);
        slog::info << "Compile model took " << double_to_string(duration_ms) << " ms" << slog::endl;
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        std::map<std::string, ov::Tensor> input_tensor_map;
        const auto& inputs = model->inputs();
        int input_ids_len = prompt.size();
        slog::info << "input_ids_len: " << input_ids_len << slog::endl;
        int eos_token_id = llama_tokenizer.get_llama_token_eos();
        bool first_iteration = true;
        const float *next_token_logits_buffer = nullptr;
        int n_vocab = llama_tokenizer.get_llama_n_vocab();
        std::vector<int> output_ids;
        int next_tokens = -1;
        std::vector<llama_token_data> candidates;
        int64_t* input_ids = (int64_t *)malloc(input_ids_len * sizeof(int64_t));
        int64_t* attention_mask = (int64_t *)malloc(input_ids_len * sizeof(int64_t));
        startTime = Time::now();
        while (1) {
            for (auto& input : model->inputs()) {
                auto input_name = input.get_any_name();
                if (input_name.find("past_key_values") != std::string::npos) {
                    if (first_iteration) {
                        int kv_shape = input.get_partial_shape()[1].get_length();
                        float *past_key_values = (float *)calloc(1 * kv_shape * 128 * 0, sizeof(float));
                        infer_request.set_tensor(input_name, ov::Tensor(input.get_element_type(), {1, (long unsigned int)kv_shape, 0, 128}, past_key_values));
                        free(past_key_values);
                    } else {
                        std::string output_name = "present" + input_name.substr(15);
                        infer_request.set_tensor(input_name, infer_request.get_tensor(output_name));                            
                    }
                } else if (input_name.find("input_ids") != std::string::npos) {
                    if (first_iteration) {
                        for (int i = 0; i < input_ids_len; i++) {
                            input_ids[i] = prompt[i];
                        }
                    } else {
                        input_ids[0] = next_tokens;
                    }
                    infer_request.set_tensor(input_name, ov::Tensor(input.get_element_type(), {1, (long unsigned int)input_ids_len}, input_ids));
                } else if (input_name.find("attention_mask") != std::string::npos) {
                    for (int i = 0; i < input_ids_len; i++) {
                        attention_mask[i] = 1;
                    }
                    infer_request.set_tensor(input_name, ov::Tensor(input.get_element_type(), {1, (long unsigned int)input_ids_len}, attention_mask));
                }
            }
            infer_request.start_async();
            infer_request.wait();
            auto output = infer_request.get_tensor("logits");
            auto output_last_aix_size = output.get_shape()[2];
            if (first_iteration) {
                ov::Tensor next_token_logits{output, {0, (long unsigned int)input_ids_len - 1, 0}, {1, (long unsigned int)input_ids_len, output_last_aix_size}};
                next_token_logits_buffer = next_token_logits.data<const float>();
                first_iteration = false;
                input_ids_len = 1;
                duration_ms = get_duration_ms_till_now(startTime);
                slog::info << "Generate 1st token took " <<  double_to_string(duration_ms) << " ms" << slog::endl;
            } else {
                next_token_logits_buffer = output.data<const float>();
            }
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{token_id, next_token_logits_buffer[token_id], 0.0f});
            }
            llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};
            next_tokens = llama_token_greedy(&candidates_p);
            candidates.clear();
            // next_tokens = std::max_element(next_token_logits_buffer, next_token_logits_buffer + output_last_aix_size) - next_token_logits_buffer;
            output_ids.push_back(next_tokens);
            if (((int)output_ids.size() == max_sequence_length) || next_tokens == eos_token_id) {
                break;
            }
        }
        duration_ms = get_duration_ms_till_now(startTime);
        slog::info << "Generate " << output_ids.size() << " new tokens took " << double_to_string(duration_ms) << " ms" << slog::endl;
        slog::info << "Lantacy: " << double_to_string(duration_ms / output_ids.size()) << " ms/token" << slog::endl;
        for(auto id : output_ids) {
            std::cout << llama_tokenizer.tokenizer_decode(id);
        }
        std::cout << std::endl;
        free(input_ids);
        free(attention_mask);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
