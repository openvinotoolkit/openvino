// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Greedy decode benchmark for the Bonsai-27B (qwen3_5) OpenVINO export. The
// decoder is the VLM language model: it consumes inputs_embeds from a separate
// embedding model and rank-3 mrope position_ids.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <openvino/openvino.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace {

int64_t argmax_last(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();  // [B, L, V]
    const size_t vocab = shape[2];
    const float* row = logits.data<float>() + (shape[1] - 1) * vocab;
    return static_cast<int64_t>(std::distance(row, std::max_element(row, row + vocab)));
}

std::vector<int64_t> parse_ids(const std::string& csv) {
    std::vector<int64_t> ids;
    std::stringstream stream(csv);
    std::string value;
    while (std::getline(stream, value, ','))
        if (!value.empty())
            ids.push_back(std::stoll(value));
    return ids;
}

void set_mask(ov::InferRequest& request, size_t length) {
    ov::Tensor mask(ov::element::i64, ov::Shape{1, length});
    std::fill_n(mask.data<int64_t>(), length, int64_t{1});
    request.set_tensor("attention_mask", mask);
}

// Text-only mrope: every section carries the same linear text position.
void set_positions(ov::InferRequest& request, int64_t start, size_t length) {
    ov::Tensor positions(ov::element::i64, ov::Shape{4, 1, length});
    auto* data = positions.data<int64_t>();
    for (size_t section = 0; section < 4; ++section)
        for (size_t i = 0; i < length; ++i)
            data[section * length + i] = start + static_cast<int64_t>(i);
    request.set_tensor("position_ids", positions);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: bench_llm_27b <language_model.xml> <text_embeddings.xml> <device>"
                     " [max_new_tokens] [comma-separated input ids]\n";
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

    try {
        ov::Core core;
        // The linear-attention (GatedDeltaNet) state math overflows in f16, so allow overriding.
        const char* precision = std::getenv("BENCH_PRECISION");
        ov::AnyMap config{{"INFERENCE_PRECISION_HINT", precision ? precision : "f16"}};
        if (std::getenv("BENCH_PROFILE"))
            config[ov::enable_profiling.name()] = true;
        if (const char* cache = std::getenv("OV_CACHE_DIR"))
            core.set_property(ov::cache_dir(cache));
        // The GPU plugin rejects unbounded dynamic dims here, so give the sequence an upper bound.
        const int max_len = std::getenv("BENCH_MAX_LEN") ? std::stoi(std::getenv("BENCH_MAX_LEN")) : 4096;
        const ov::Dimension seq(1, max_len);
        // BENCH_STATIC_DECODE pins the token dimension to 1 so the graph's
        // ShapeOf/Range/Gather plumbing folds away; the prompt is then fed one
        // token at a time. attention_mask still has to grow with the context.
        const bool static_decode = std::getenv("BENCH_STATIC_DECODE") != nullptr;
        const ov::Dimension tok = static_decode ? ov::Dimension(1) : seq;

        const auto compile_start = std::chrono::high_resolution_clock::now();
        auto embed_model = core.read_model(embed_path);
        embed_model->reshape({{embed_model->input(0).get_any_name(), ov::PartialShape{1, tok}}});
        auto embed = core.compile_model(embed_model, device, config);

        auto lm_model = core.read_model(lm_path);
        lm_model->reshape({{"attention_mask", ov::PartialShape{1, seq}},
                           {"inputs_embeds", ov::PartialShape{1, tok, 5120}},
                           {"position_ids", ov::PartialShape{4, 1, tok}},
                           {"beam_idx", ov::PartialShape{1}}});
        auto lm = core.compile_model(lm_model, device, config);
        const auto compile_end = std::chrono::high_resolution_clock::now();
        std::cout << "compile " << std::chrono::duration<double>(compile_end - compile_start).count() << " s\n";

        for (const auto& node : lm.get_runtime_model()->get_ordered_ops()) {
            const auto& rt = node->get_rt_info();
            const auto it = rt.find("primitiveType");
            if (it != rt.end() && it->second.as<std::string>().find("ternocl_int2") != std::string::npos) {
                std::cout << "ternocl int2 impl in use (first hit: " << node->get_friendly_name() << ")\n";
                break;
            }
        }

        auto embed_request = embed.create_infer_request();
        auto lm_request = lm.create_infer_request();
        lm_request.reset_state();
        {
            ov::Tensor beam(ov::element::i32, ov::Shape{1});
            beam.data<int32_t>()[0] = 0;
            lm_request.set_tensor("beam_idx", beam);
        }

        auto embed_tokens = [&](const std::vector<int64_t>& ids) {
            ov::Tensor input(ov::element::i64, ov::Shape{1, ids.size()});
            std::copy(ids.begin(), ids.end(), input.data<int64_t>());
            embed_request.set_tensor("input", input);
            embed_request.infer();
            lm_request.set_tensor("inputs_embeds", embed_request.get_tensor("inputs_embeds"));
        };

        const bool debug = std::getenv("BENCH_DEBUG") != nullptr;
        auto describe = [&](const char* label, const ov::Tensor& t) {
            if (!debug)
                return;
            const float* p = t.data<const float>();
            const size_t n = t.get_size();
            double lo = p[0], hi = p[0], sum = 0.0;
            size_t nonfinite = 0;
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(p[i])) {
                    ++nonfinite;
                    continue;
                }
                lo = std::min<double>(lo, p[i]);
                hi = std::max<double>(hi, p[i]);
                sum += p[i];
            }
            std::cerr << "[dbg] " << label << " shape=" << t.get_shape() << " min=" << lo << " max=" << hi << " mean=" << sum / static_cast<double>(n)
                      << " nonfinite=" << nonfinite << '\n';
        };

        const auto prefill_start = std::chrono::high_resolution_clock::now();
        if (static_decode) {
            // One token at a time, since the graph only accepts a single token.
            for (size_t i = 0; i < prompt.size(); ++i) {
                embed_tokens({prompt[i]});
                set_mask(lm_request, i + 1);
                set_positions(lm_request, static_cast<int64_t>(i), 1);
                lm_request.infer();
            }
        } else {
            embed_tokens(prompt);
            describe("inputs_embeds", embed_request.get_tensor("inputs_embeds"));
            set_mask(lm_request, prompt.size());
            set_positions(lm_request, 0, prompt.size());
            lm_request.infer();
        }
        const auto prefill_end = std::chrono::high_resolution_clock::now();
        describe("logits", lm_request.get_tensor("logits"));

        int64_t next = argmax_last(lm_request.get_tensor("logits"));
        std::vector<int64_t> generated{next};

        const bool no_eos = std::getenv("BENCH_NO_EOS") != nullptr;
        int decoded = 0;
        const auto decode_start = std::chrono::high_resolution_clock::now();
        for (int step = 1; step < max_new; ++step) {
            const auto past = prompt.size() + static_cast<size_t>(step) - 1;
            embed_tokens({next});
            set_mask(lm_request, past + 1);
            set_positions(lm_request, static_cast<int64_t>(past), 1);
            lm_request.infer();
            next = argmax_last(lm_request.get_tensor("logits"));
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
            // Counters are from the last decode step, so this is a per-token profile.
            std::map<std::string, std::pair<double, size_t>> by_type;
            double total = 0;
            for (const auto& p : lm_request.get_profiling_info()) {
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
            for (size_t i = 0; i < v.size() && i < 16; ++i)
                std::cout << "  " << std::setw(28) << std::left << v[i].first << std::setw(10) << std::right << v[i].second.first
                          << " us  n=" << v[i].second.second << "  (" << (100.0 * v[i].second.first / total) << "%)\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "27b decode failed: " << error.what() << '\n';
        return 2;
    }
}
