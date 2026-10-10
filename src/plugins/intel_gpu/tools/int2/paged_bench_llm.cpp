// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <openvino/openvino.hpp>
#include <openvino/pass/sdpa_to_paged_attention.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<int64_t> parse_ids(const std::string& csv) {
    std::vector<int64_t> ids;
    std::stringstream stream(csv);
    std::string value;
    while (std::getline(stream, value, ',')) {
        if (!value.empty())
            ids.push_back(std::stoll(value));
    }
    return ids;
}

int64_t argmax_last(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    const size_t vocab = shape.back();
    const float* values = logits.data<const float>() + logits.get_size() - vocab;
    return static_cast<int64_t>(std::distance(values, std::max_element(values, values + vocab)));
}

void set_i64(ov::InferRequest& request, const char* name, const std::vector<int64_t>& values) {
    ov::Tensor tensor(ov::element::i64, ov::Shape{values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int64_t>());
    request.set_tensor(name, tensor);
}

void set_i32(ov::InferRequest& request, const char* name, const std::vector<int32_t>& values) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int32_t>());
    request.set_tensor(name, tensor);
}

void set_i32_scalar(ov::InferRequest& request, const char* name, int32_t value) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{});
    tensor.data<int32_t>()[0] = value;
    request.set_tensor(name, tensor);
}

void set_active_blocks(ov::InferRequest& request, int32_t token_count, int32_t block_size) {
    const int32_t active_block_count = (token_count + block_size - 1) / block_size;
    std::vector<int32_t> block_indices(active_block_count);
    for (int32_t index = 0; index < active_block_count; ++index)
        block_indices[index] = index;
    set_i32(request, "block_indices", block_indices);
    set_i32(request, "block_indices_begins", {0, active_block_count});
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: paged_bench_llm <model.xml> <device> [max_new_tokens] [comma-separated input ids]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string device = argv[2];
    const int max_new_tokens = argc > 3 ? std::stoi(argv[3]) : 256;
    const std::string ids_csv = argc > 4 ? argv[4] : "151644,872,198,840,20772,7249,73667,304,1378,22870,13";
    const auto prompt = parse_ids(ids_csv);
    constexpr int32_t max_context_len = 512;
    constexpr int32_t block_size = 16;
    constexpr int32_t num_blocks = max_context_len / block_size;

    if (prompt.empty() || static_cast<int32_t>(prompt.size()) + max_new_tokens > max_context_len) {
        std::cerr << "prompt plus generated tokens must fit the " << max_context_len << " token cache\n";
        return 1;
    }

    try {
        ov::Core core;
        auto model = core.read_model(model_path);
        ov::pass::SDPAToPagedAttention paged_attention_pass;
        paged_attention_pass.run_on_model(model);

        const auto compile_start = std::chrono::high_resolution_clock::now();
        ov::AnyMap compile_options{{"INFERENCE_PRECISION_HINT", "f16"}};
        if (std::getenv("PAGED_PROFILE"))
            compile_options[ov::enable_profiling.name()] = true;
        auto compiled = core.compile_model(model, device, compile_options);
        const auto compile_end = std::chrono::high_resolution_clock::now();
        std::cout << "compile " << std::chrono::duration<double>(compile_end - compile_start).count() << " s\n";

        auto request = compiled.create_infer_request();
        auto context = compiled.get_context();

        // Layer count and cache geometry vary per model, and the plugin's
        // ConvertPagedAttnInputs pass has already fixed the shapes, so read both
        // off the compiled model rather than assuming one architecture.
        int32_t layer_count = 0;
        for (const auto& port : compiled.inputs()) {
            for (const auto& name : port.get_names()) {
                if (name.rfind("key_cache.", 0) == 0)
                    ++layer_count;
            }
        }
        std::cout << "paged layers: " << layer_count << '\n';

        std::vector<ov::RemoteTensor> caches;
        caches.reserve(layer_count * 2);
        auto bind = [&](const std::string& name) {
            const auto port = compiled.input(name);
            const auto partial = port.get_partial_shape();
            ov::Shape shape(partial.size());
            for (size_t i = 0; i < partial.size(); ++i)
                shape[i] = partial[i].is_dynamic() ? static_cast<size_t>(num_blocks) : static_cast<size_t>(partial[i].get_length());
            caches.emplace_back(context.create_tensor(port.get_element_type(), shape));
            request.set_tensor(name, caches.back());
        };
        for (int32_t layer = 0; layer < layer_count; ++layer) {
            bind("key_cache." + std::to_string(layer));
            bind("value_cache." + std::to_string(layer));
        }

        set_i32_scalar(request, "max_context_len", max_context_len);

        std::vector<int64_t> position_ids(prompt.size());
        for (size_t index = 0; index < prompt.size(); ++index)
            position_ids[index] = static_cast<int64_t>(index);
        set_i64(request, "input_ids", prompt);
        set_i64(request, "position_ids", position_ids);
        set_i32(request, "past_lens", {0});
        set_i32(request, "subsequence_begins", {0, static_cast<int32_t>(prompt.size())});
        set_active_blocks(request, static_cast<int32_t>(prompt.size()), block_size);
        const auto prefill_start = std::chrono::high_resolution_clock::now();
        request.infer();

        int64_t next = argmax_last(request.get_tensor("logits"));
        const auto prefill_end = std::chrono::high_resolution_clock::now();
        std::vector<int64_t> generated{next};
        const auto decode_start = std::chrono::high_resolution_clock::now();
        for (int token_index = 1; token_index < max_new_tokens; ++token_index) {
            const int32_t past_length = static_cast<int32_t>(prompt.size()) + token_index - 1;
            set_i64(request, "input_ids", {next});
            set_i64(request, "position_ids", {past_length});
            set_i32(request, "past_lens", {past_length});
            set_i32(request, "subsequence_begins", {0, 1});
            set_active_blocks(request, past_length + 1, block_size);
            request.infer();
            next = argmax_last(request.get_tensor("logits"));
            generated.push_back(next);
        }
        const auto decode_end = std::chrono::high_resolution_clock::now();
        const double decode_seconds = std::chrono::duration<double>(decode_end - decode_start).count();

        if (const char* exec_graph_path = std::getenv("PAGED_EXEC_GRAPH_PATH"))
            ov::serialize(compiled.get_runtime_model(), exec_graph_path);

        std::cout << "TTFT (prefill) : " << std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count() << " ms\n";
        std::cout << "decode         : " << max_new_tokens - 1 << " tokens in " << decode_seconds << " s = " << (max_new_tokens - 1) / decode_seconds
                  << " tok/s\n";
        std::cout << "generated_ids  =";
        for (const auto id : generated)
            std::cout << id << ',';
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "paged decode failed: " << error.what() << '\n';
        return 2;
    }
}
