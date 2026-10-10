// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Greedy decode benchmark for the Bonsai-27B (qwen3_5) export converted with
// SDPAToPagedAttention. Versus the stateful path this replaces the 9-op
// causal-conv1d chain with one PagedCausalConv1D per linear-attention layer and
// the Loop-derived GatedDeltaNet with PagedGatedDeltaNet.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <openvino/openvino.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/sdpa_to_paged_attention.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int32_t kFullAttnLayers = 16;
constexpr int32_t kLinearAttnLayers = 48;
constexpr int32_t kKVHeads = 4;
constexpr int32_t kHeadDim = 256;
constexpr int32_t kConvChannels = 10240;
constexpr int32_t kConvKernel = 4;
constexpr int32_t kGdnValueHeads = 48;
constexpr int32_t kGdnKeyDim = 128;
constexpr int32_t kGdnValueDim = 128;
constexpr int32_t kHidden = 5120;

std::vector<int64_t> parse_ids(const std::string& csv) {
    std::vector<int64_t> ids;
    std::stringstream stream(csv);
    std::string value;
    while (std::getline(stream, value, ','))
        if (!value.empty())
            ids.push_back(std::stoll(value));
    return ids;
}

void set_i32(ov::InferRequest& request, const std::string& name, const std::vector<int32_t>& values) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int32_t>());
    request.set_tensor(name, tensor);
}

void set_i32_scalar(ov::InferRequest& request, const std::string& name, int32_t value) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{});
    *tensor.data<int32_t>() = value;
    request.set_tensor(name, tensor);
}

// Full-attention KV blocks; the linear-attention state is per-sequence and uses
// a single slot, so it is not paged the same way.
void set_active_blocks(ov::InferRequest& request, int32_t token_count, int32_t block_size) {
    const int32_t active = (token_count + block_size - 1) / block_size;
    std::vector<int32_t> block_indices(active);
    for (int32_t i = 0; i < active; ++i)
        block_indices[i] = i;
    set_i32(request, "block_indices", block_indices);
    set_i32(request, "block_indices_begins", {0, active});
}

// The kernel reads the state from block_indices[begin] and writes the updated
// state to block_indices[begin + 1], so the two slots must swap every call.
// cache_interval <= 0 keeps that to a single write per call.
constexpr int32_t kLinearAttnSlots = 2;

void set_linear_attn_meta(ov::InferRequest& request, int32_t past_length, int32_t read_slot) {
    set_i32(request, "la.past_lens", {past_length});
    set_i32(request, "la.block_indices", {read_slot, 1 - read_slot});
    set_i32(request, "la.block_indices_begins", {0, kLinearAttnSlots});
    set_i32(request, "la.cache_interval", {0});
}

int64_t argmax_last(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    const size_t vocab = shape.back();
    size_t rows = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i)
        rows *= shape[i];
    const float* row = logits.data<float>() + (rows - 1) * vocab;
    return static_cast<int64_t>(std::distance(row, std::max_element(row, row + vocab)));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: paged_bench_llm_27b <paged_language_model.xml> <text_embeddings.xml>"
                     " <device> [max_new_tokens] [comma-separated input ids]\n";
        return 1;
    }
    const std::string lm_path = argv[1];
    const std::string embed_path = argv[2];
    const std::string device = argv[3];
    const int max_new = argc > 4 ? std::stoi(argv[4]) : 256;
    const auto prompt = parse_ids(argc > 5 ? argv[5] : "151644,872,198");
    if (prompt.empty()) {
        std::cerr << "empty prompt\n";
        return 1;
    }

    const int32_t block_size = std::getenv("BENCH_BLOCK_SIZE") ? std::stoi(std::getenv("BENCH_BLOCK_SIZE")) : 16;
    const int32_t max_context_len = std::getenv("BENCH_MAX_LEN") ? std::stoi(std::getenv("BENCH_MAX_LEN")) : 512;
    const int32_t num_blocks = (max_context_len + block_size - 1) / block_size;
    if (static_cast<int32_t>(prompt.size()) + max_new > max_context_len) {
        std::cerr << "prompt plus generated tokens must fit " << max_context_len << " tokens\n";
        return 1;
    }

    try {
        ov::Core core;
        const char* precision = std::getenv("BENCH_PRECISION");
        ov::AnyMap config{{"INFERENCE_PRECISION_HINT", precision ? precision : "f16"}};
        if (std::getenv("BENCH_PROFILE"))
            config[ov::enable_profiling.name()] = true;
        if (const char* cache = std::getenv("OV_CACHE_DIR"))
            core.set_property(ov::cache_dir(cache));

        const auto compile_start = std::chrono::high_resolution_clock::now();
        auto embed_model = core.read_model(embed_path);
        embed_model->reshape({{embed_model->input(0).get_any_name(), ov::PartialShape{1, -1}}});
        auto embed = core.compile_model(embed_model, device, config);
        // Converted in-process: the paged ops are extension-opset and cannot be
        // deserialized back from a saved IR.
        auto lm_model = core.read_model(lm_path);
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::SDPAToPagedAttention>();
        manager.run_passes(lm_model);
        auto lm = core.compile_model(lm_model, device, config);
        const auto compile_end = std::chrono::high_resolution_clock::now();
        std::cout << "compile " << std::chrono::duration<double>(compile_end - compile_start).count() << " s\n";

        if (std::getenv("BENCH_RTOPS")) {
            std::map<std::string, size_t> by_type;
            for (const auto& node : lm.get_runtime_model()->get_ordered_ops()) {
                const auto& rt = node->get_rt_info();
                const auto it = rt.find("layerType");
                by_type[it != rt.end() ? it->second.as<std::string>() : node->get_type_name()]++;
            }
            size_t total = 0;
            for (const auto& e : by_type)
                total += e.second;
            std::vector<std::pair<std::string, size_t>> v(by_type.begin(), by_type.end());
            std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
            std::cout << "runtime graph: " << total << " nodes\n";
            for (size_t i = 0; i < v.size() && i < 24; ++i)
                std::cout << "   " << std::setw(30) << std::left << v[i].first << v[i].second << '\n';
        }

        auto embed_request = embed.create_infer_request();
        auto request = lm.create_infer_request();

        // Held for the whole run: the kernels update these in place. Shapes and
        // precisions are decided by the plugin's ConvertPagedAttnInputs pass, so
        // take them from the compiled model and only fill in the block count.
        // They must be device tensors: a host tensor is re-uploaded before every
        // infer, which would discard the in-place state update.
        std::vector<ov::Tensor> caches;
        const bool host_caches = std::getenv("BENCH_HOST_CACHES") != nullptr;
        ov::RemoteContext context = lm.get_context();
        auto bind = [&](const std::string& name, size_t blocks) {
            const auto port = lm.input(name);
            const auto partial = port.get_partial_shape();
            ov::Shape shape(partial.size());
            for (size_t i = 0; i < partial.size(); ++i)
                shape[i] = partial[i].is_dynamic() ? blocks : static_cast<size_t>(partial[i].get_length());
            ov::Tensor zeros(port.get_element_type(), shape);
            std::memset(zeros.data(), 0, zeros.get_byte_size());
            if (host_caches) {
                caches.emplace_back(std::move(zeros));
            } else {
                auto remote = context.create_tensor(port.get_element_type(), shape);
                remote.copy_from(zeros);
                caches.emplace_back(std::move(remote));
            }
            request.set_tensor(name, caches.back());
        };
        for (int32_t layer = 0; layer < kFullAttnLayers; ++layer) {
            bind("key_cache." + std::to_string(layer), static_cast<size_t>(num_blocks));
            bind("value_cache." + std::to_string(layer), static_cast<size_t>(num_blocks));
        }
        for (int32_t layer = 0; layer < kLinearAttnLayers; ++layer) {
            bind("conv_state_table." + std::to_string(layer), kLinearAttnSlots);
            bind("gated_delta_state_table." + std::to_string(layer), kLinearAttnSlots);
        }
        set_i32_scalar(request, "max_context_len", max_context_len);
        if (std::getenv("BENCH_MEMINFO")) {
            size_t bytes = 0;
            for (const auto& t : caches)
                bytes += t.get_byte_size();
            std::cout << "caches: " << caches.size() << " tensors, " << bytes / (1024 * 1024) << " MiB, " << (host_caches ? "host" : "device") << '\n';
        }

        // inputs_embeds is [tokens, hidden] on the paged path, not [1, tokens, hidden].
        auto feed_embeds = [&](const std::vector<int64_t>& ids) {
            ov::Tensor input(ov::element::i64, ov::Shape{1, ids.size()});
            std::copy(ids.begin(), ids.end(), input.data<int64_t>());
            embed_request.set_input_tensor(input);
            embed_request.infer();
            const auto out = embed_request.get_output_tensor();
            ov::Tensor embeds(lm.input("inputs_embeds").get_element_type(), ov::Shape{ids.size(), kHidden});
            std::memcpy(embeds.data(), out.data(), embeds.get_byte_size());
            request.set_tensor("inputs_embeds", embeds);
        };
        auto set_positions = [&](int64_t start, size_t length) {
            ov::Tensor positions(ov::element::i64, ov::Shape{4, length});
            auto* data = positions.data<int64_t>();
            for (size_t section = 0; section < 4; ++section)
                for (size_t i = 0; i < length; ++i)
                    data[section * length + i] = start + static_cast<int64_t>(i);
            request.set_tensor("position_ids", positions);
        };

        const auto prefill_start = std::chrono::high_resolution_clock::now();
        int32_t read_slot = 0;
        // A whole-prompt prefill needs an ESIMD scratch surface that does not
        // fit next to the weights on an integrated GPU; chunking bounds it.
        const int32_t chunk = std::getenv("BENCH_PREFILL_CHUNK") ? std::stoi(std::getenv("BENCH_PREFILL_CHUNK")) : static_cast<int32_t>(prompt.size());
        for (size_t start = 0; start < prompt.size(); start += static_cast<size_t>(chunk)) {
            const size_t len = std::min(static_cast<size_t>(chunk), prompt.size() - start);
            const std::vector<int64_t> part(prompt.begin() + static_cast<long>(start), prompt.begin() + static_cast<long>(start + len));
            feed_embeds(part);
            set_positions(static_cast<int64_t>(start), len);
            set_i32(request, "past_lens", {static_cast<int32_t>(start)});
            set_i32(request, "subsequence_begins", {0, static_cast<int32_t>(len)});
            set_active_blocks(request, static_cast<int32_t>(start + len), block_size);
            set_linear_attn_meta(request, static_cast<int32_t>(start), read_slot);
            request.infer();
            read_slot = 1 - read_slot;
        }
        const auto prefill_end = std::chrono::high_resolution_clock::now();

        int64_t next = argmax_last(request.get_output_tensor());
        std::vector<int64_t> generated{next};

        const bool no_eos = std::getenv("BENCH_NO_EOS") != nullptr;
        int decoded = 0;
        const auto decode_start = std::chrono::high_resolution_clock::now();
        for (int step = 1; step < max_new; ++step) {
            const int32_t past = static_cast<int32_t>(prompt.size()) + step - 1;
            feed_embeds({next});
            set_positions(past, 1);
            set_i32(request, "past_lens", {past});
            set_i32(request, "subsequence_begins", {0, 1});
            set_active_blocks(request, past + 1, block_size);
            set_linear_attn_meta(request, past, read_slot);
            request.infer();
            read_slot = 1 - read_slot;
            next = argmax_last(request.get_output_tensor());
            generated.push_back(next);
            ++decoded;
            if (!no_eos && (next == 248046 || next == 248044))
                break;
        }
        const auto decode_end = std::chrono::high_resolution_clock::now();
        const double decode_seconds = std::chrono::duration<double>(decode_end - decode_start).count();

        std::cout << "TTFT (prefill) : " << std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count() << " ms\n";
        std::cout << "decode         : " << decoded << " tokens in " << decode_seconds << " s = " << decoded / decode_seconds << " tok/s\n";
        std::cout << "generated_ids  =";
        for (const auto id : generated)
            std::cout << id << ',';
        std::cout << '\n';

        if (std::getenv("BENCH_PROFILE")) {
            std::map<std::string, std::pair<double, size_t>> by_type;
            double total = 0;
            for (const auto& p : request.get_profiling_info()) {
                if (p.status == ov::ProfilingInfo::Status::NOT_RUN)
                    continue;
                const double us = static_cast<double>(p.real_time.count());
                auto& e = by_type[p.node_type];
                e.first += us;
                e.second += 1;
                total += us;
            }
            std::vector<std::pair<std::string, std::pair<double, size_t>>> v(by_type.begin(), by_type.end());
            std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
                return a.second.first > b.second.first;
            });
            std::cout << "per-token profile (total " << total << " us):\n";
            for (size_t i = 0; i < v.size() && i < 20; ++i)
                std::cout << "  " << std::setw(28) << std::left << v[i].first << std::setw(10) << std::right << v[i].second.first
                          << " us  n=" << v[i].second.second << "  (" << (100.0 * v[i].second.first / total) << "%)\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "paged 27b decode failed: " << error.what() << '\n';
        return 1;
    }
}
