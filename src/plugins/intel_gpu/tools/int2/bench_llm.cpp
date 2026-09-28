// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Greedy decode benchmark for a stateful OpenVINO LLM, used to measure the
// TernOCL int2 FullyConnected impl end to end.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
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
    const float* p = logits.data<float>() + (shape[1] - 1) * vocab;
    int64_t best = 0;
    float best_v = p[0];
    for (size_t i = 1; i < vocab; ++i) {
        if (p[i] > best_v) {
            best_v = p[i];
            best = static_cast<int64_t>(i);
        }
    }
    return best;
}

void set_i64(ov::InferRequest& req, const char* name, const std::vector<int64_t>& v, size_t rows = 1) {
    ov::Tensor t(ov::element::i64, ov::Shape{rows, v.size() / rows});
    std::copy(v.begin(), v.end(), t.data<int64_t>());
    req.set_tensor(name, t);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: bench_llm <model.xml> <device> [max_new_tokens] [comma-separated input ids]\n";
        return 1;
    }
    std::string model_path = argv[1];
    std::string device = argc > 2 ? argv[2] : "GPU";
    const int max_new = argc > 3 ? std::stoi(argv[3]) : 128;
    std::string ids_csv = argc > 4 ? argv[4]
                                   : "151644,872,198,840,20772,7249,73667,304,1378,22870,13,"
                                     "151645,198,151644,77091,198,151667,271,151668,271";

    std::vector<int64_t> prompt;
    {
        std::stringstream ss(ids_csv);
        std::string tokstr;
        while (std::getline(ss, tokstr, ','))
            if (!tokstr.empty())
                prompt.push_back(std::stoll(tokstr));
    }

    ov::Core core;
    std::cerr << "[bench] reading+compiling " << model_path << " on " << device << std::endl;
    const auto t_load0 = std::chrono::high_resolution_clock::now();
    const bool do_prof = std::getenv("BENCH_PROFILE") != nullptr;
    ov::AnyMap cfg{{"INFERENCE_PRECISION_HINT", "f16"}};
    if (do_prof)
        cfg[ov::enable_profiling.name()] = true;
    auto compiled = core.compile_model(model_path, device, cfg);
    const auto t_load1 = std::chrono::high_resolution_clock::now();
    std::cout << "compile " << std::chrono::duration<double>(t_load1 - t_load0).count() << " s\n";

    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
        const auto& rt = node->get_rt_info();
        auto it = rt.find("primitiveType");
        if (it != rt.end() && it->second.as<std::string>().find("ternocl_int2") != std::string::npos) {
            std::cout << "ternocl int2 impl in use (first hit: " << node->get_friendly_name() << ")\n";
            break;
        }
    }

    auto req = compiled.create_infer_request();
    req.reset_state();

    std::vector<int64_t> mask(prompt.size(), 1), pos(prompt.size());
    for (size_t i = 0; i < prompt.size(); ++i)
        pos[i] = static_cast<int64_t>(i);
    set_i64(req, "input_ids", prompt);
    set_i64(req, "attention_mask", mask);
    set_i64(req, "position_ids", pos);
    {
        ov::Tensor b(ov::element::i32, ov::Shape{1});
        b.data<int32_t>()[0] = 0;
        req.set_tensor("beam_idx", b);
    }

    const auto t_pre0 = std::chrono::high_resolution_clock::now();
    req.infer();
    const auto t_pre1 = std::chrono::high_resolution_clock::now();
    const double ttft_ms = std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count();

    int64_t next = argmax_last(req.get_tensor("logits"));
    std::vector<int64_t> generated{next};

    const auto t_dec0 = std::chrono::high_resolution_clock::now();
    int decoded = 0;
    for (int i = 1; i < max_new; ++i) {
        mask.push_back(1);
        set_i64(req, "input_ids", std::vector<int64_t>{next});
        set_i64(req, "attention_mask", mask);
        set_i64(req, "position_ids", std::vector<int64_t>{static_cast<int64_t>(prompt.size() + i - 1)});
        req.infer();
        // BENCH_DUMP_LOGITS=<step>:<path> writes raw logits for that decode step,
        // so two builds can be compared numerically rather than by token.
        if (const char* d = std::getenv("BENCH_DUMP_LOGITS")) {
            const std::string spec(d);
            const auto colon = spec.find(':');
            if (colon != std::string::npos && std::stoi(spec.substr(0, colon)) == i) {
                auto lt = req.get_tensor("logits");
                std::ofstream f(spec.substr(colon + 1), std::ios::binary);
                f.write(reinterpret_cast<const char*>(lt.data<float>()), lt.get_size() * sizeof(float));
            }
        }
        next = argmax_last(req.get_tensor("logits"));
        generated.push_back(next);
        ++decoded;
        // BENCH_NO_EOS keeps decoding so every run does identical work.
        if ((next == 151645 || next == 151643) && std::getenv("BENCH_NO_EOS") == nullptr)
            break;
    }
    const auto t_dec1 = std::chrono::high_resolution_clock::now();
    const double dec_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();

    std::cout << "TTFT (prefill) : " << ttft_ms << " ms\n";
    if (do_prof) {
        // Aggregated over the last decode step only; SYCL impls report 0 because
        // to_ocl_event() yields no event on an in-order queue.
        std::map<std::string, double> by_type;
        std::map<std::string, size_t> count_by_type;
        double total = 0.0;
        for (const auto& p : req.get_profiling_info()) {
            const double us = static_cast<double>(p.real_time.count());
            by_type[p.node_type] += us;
            count_by_type[p.node_type] += 1;
            total += us;
        }
        std::vector<std::pair<double, std::string>> v;
        for (const auto& kv : by_type)
            v.emplace_back(kv.second, kv.first);
        std::sort(v.rbegin(), v.rend());
        std::cout << "[prof] total reported " << total << " us over one decode step\n";
        for (size_t i = 0; i < v.size() && i < 15; ++i)
            std::cout << "[prof] " << v[i].second << " " << v[i].first << " us over " << count_by_type[v[i].second] << " calls ("
                      << v[i].first / count_by_type[v[i].second] << " us each)\n";
    }
    std::cout << "decode         : " << decoded << " tokens in " << dec_s << " s = " << decoded / dec_s << " tok/s\n";
    std::cout << "generated_ids  =";
    for (auto t : generated)
        std::cout << t << ",";
    std::cout << "\n";
    return 0;
}
