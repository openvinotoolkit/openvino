// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Minimal test: compare TENSOR_PARALLEL vs GPU inference on any stateful LLM.
//
// Usage:
//   ./tp_test <model.xml> [tp_degree]
//
// Example:
//   ./tp_test /path/to/TinyLlama-1.1B-int4/openvino_model.xml 2

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

// Fill tensor with a fixed value based on element type.
static void fill_input(ov::Tensor& tensor, const std::string& name) {
    auto type = tensor.get_element_type();
    auto shape = tensor.get_shape();
    size_t total = tensor.get_size();

    if (type == ov::element::i64) {
        auto* p = tensor.data<int64_t>();
        if (name.find("input_ids") != std::string::npos) {
            // Dummy token sequence: [1, 15043, 29892, 920] ("Hello, world" approx)
            std::vector<int64_t> tokens = {1, 15043, 29892, 920};
            for (size_t i = 0; i < total; ++i)
                p[i] = tokens[i % tokens.size()];
        } else if (name.find("attention_mask") != std::string::npos) {
            for (size_t i = 0; i < total; ++i)
                p[i] = 1;
        } else if (name.find("position_ids") != std::string::npos) {
            for (size_t i = 0; i < total; ++i)
                p[i] = static_cast<int64_t>(i);
        } else {
            for (size_t i = 0; i < total; ++i)
                p[i] = 0;
        }
    } else if (type == ov::element::i32) {
        auto* p = tensor.data<int32_t>();
        for (size_t i = 0; i < total; ++i)
            p[i] = 0;
    } else if (type == ov::element::f32) {
        auto* p = tensor.data<float>();
        for (size_t i = 0; i < total; ++i)
            p[i] = 0.0f;
    }
}

struct InferResult {
    std::vector<float> logits;
    ov::Shape shape;
    double time_ms;
};

// Build a properly-shaped input tensor for a given (input_seq_len, past_len).
// past_len = 0 corresponds to a fresh prefill; past_len > 0 to a decode step
// where KV-cache already holds `past_len` tokens.
static ov::Tensor make_input_tensor(const ov::Output<const ov::Node>& input,
                                    int input_seq_len,
                                    int past_len) {
    auto name = input.get_any_name();
    auto type = input.get_element_type();
    auto ps   = input.get_partial_shape();
    ov::Shape shape;

    if (name.find("input_ids") != std::string::npos ||
        name.find("position_ids") != std::string::npos) {
        shape = {1, static_cast<size_t>(input_seq_len)};
    } else if (name.find("attention_mask") != std::string::npos) {
        // Stateful LLMs: full attention_mask covering past + current tokens.
        shape = {1, static_cast<size_t>(past_len + input_seq_len)};
    } else if (name.find("beam_idx") != std::string::npos) {
        shape = {1};
    } else {
        shape.resize(ps.rank().get_length());
        for (size_t d = 0; d < shape.size(); ++d) {
            shape[d] = ps[d].is_static() ? ps[d].get_length() : 1;
        }
    }

    ov::Tensor tensor(type, shape);
    size_t total = tensor.get_size();

    if (type == ov::element::i64) {
        auto* p = tensor.data<int64_t>();
        if (name.find("input_ids") != std::string::npos) {
            // Continue dummy token sequence across decode steps.
            std::vector<int64_t> tokens = {1, 15043, 29892, 920};
            for (size_t i = 0; i < total; ++i)
                p[i] = tokens[(past_len + static_cast<int>(i)) % tokens.size()];
        } else if (name.find("attention_mask") != std::string::npos) {
            for (size_t i = 0; i < total; ++i) p[i] = 1;
        } else if (name.find("position_ids") != std::string::npos) {
            for (size_t i = 0; i < total; ++i)
                p[i] = static_cast<int64_t>(past_len + static_cast<int>(i));
        } else {
            for (size_t i = 0; i < total; ++i) p[i] = 0;
        }
    } else if (type == ov::element::i32) {
        auto* p = tensor.data<int32_t>();
        for (size_t i = 0; i < total; ++i) p[i] = 0;
    } else if (type == ov::element::f32) {
        auto* p = tensor.data<float>();
        for (size_t i = 0; i < total; ++i) p[i] = 0.0f;
    }
    return tensor;
}

static InferResult run_inference(ov::Core& core,
                                 const std::shared_ptr<ov::Model>& model,
                                 const std::string& device,
                                 const ov::AnyMap& config,
                                 int seq_len,
                                 int iters = 1) {
    std::cout << "  Compiling on " << device << "..." << std::flush;

    auto t0 = std::chrono::steady_clock::now();
    auto compiled = core.compile_model(model, device, config);
    auto t1 = std::chrono::steady_clock::now();
    double compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(0) << compile_ms << " ms)" << std::endl;

    auto request = compiled.create_infer_request();

    // Set input shapes and fill
    for (const auto& input : compiled.inputs()) {
        auto name = input.get_any_name();
        ov::Shape shape;
        auto ps = input.get_partial_shape();

        if (name.find("input_ids") != std::string::npos ||
            name.find("attention_mask") != std::string::npos ||
            name.find("position_ids") != std::string::npos) {
            shape = {1, static_cast<size_t>(seq_len)};
        } else if (name.find("beam_idx") != std::string::npos) {
            shape = {1};
        } else {
            // Try to make it concrete with batch=1
            shape.resize(ps.rank().get_length());
            for (size_t d = 0; d < shape.size(); ++d) {
                shape[d] = ps[d].is_static() ? ps[d].get_length() : 1;
            }
        }

        ov::Tensor tensor(input.get_element_type(), shape);
        fill_input(tensor, name);
        request.set_tensor(input, tensor);
    }

    // Reset state (KV caches)
    for (auto& state : request.query_state()) {
        state.reset();
    }

    std::cout << "  Running inference (" << iters << " iter"
              << (iters > 1 ? "s" : "") << ")..." << std::flush;
    double infer_ms_first = 0.0;
    double infer_ms_min_rest = 1e18;
    double infer_ms_last = 0.0;
    double infer_ms_sum = 0.0;
    InferResult result;
    for (int it = 0; it < iters; ++it) {
        auto ti0 = std::chrono::steady_clock::now();
        request.infer();
        auto ti1 = std::chrono::steady_clock::now();
        infer_ms_last = std::chrono::duration<double, std::milli>(ti1 - ti0).count();
        infer_ms_sum += infer_ms_last;
        if (it == 0) {
            infer_ms_first = infer_ms_last;
            // iter[0] is the only iteration with guaranteed-clean state in
            // TP plugin (subsequent reset()s don't fully match a fresh run).
            auto output_tensor = request.get_output_tensor();
            auto output_shape = output_tensor.get_shape();
            const float* data = output_tensor.data<float>();
            size_t total = output_tensor.get_size();
            result.logits.assign(data, data + total);
            result.shape = output_shape;
        } else {
            infer_ms_min_rest = std::min(infer_ms_min_rest, infer_ms_last);
        }
        if (it + 1 < iters) {
            for (auto& state : request.query_state()) state.reset();
        }
    }
    // For perf: use min across iter[1..] (warm). Fall back to iter[0] if
    // only one iter was requested.
    double infer_ms = (iters > 1) ? infer_ms_min_rest : infer_ms_first;
    std::cout << " done (first=" << std::fixed << std::setprecision(1) << infer_ms_first;
    if (iters > 1) {
        std::cout << " warm_min=" << infer_ms_min_rest
                  << " avg=" << infer_ms_sum / iters
                  << " last=" << infer_ms_last;
    }
    std::cout << " ms)" << std::endl;

    result.time_ms = infer_ms;
    return result;
}

// Realistic LLM scenario: 1 prefill + N decode steps (seq_len=1) without
// resetting KV-cache between them.  Reports prefill latency, decode TPS,
// and first-token latency.  Returns prefill logits for correctness check.
static InferResult run_prefill_decode(ov::Core& core,
                                      const std::shared_ptr<ov::Model>& model,
                                      const std::string& device,
                                      const ov::AnyMap& config,
                                      int prefill_len,
                                      int gen_len) {
    std::cout << "  Compiling on " << device << "..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    auto compiled = core.compile_model(model, device, config);
    auto t1 = std::chrono::steady_clock::now();
    double compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(0) << compile_ms << " ms)" << std::endl;

    auto request = compiled.create_infer_request();
    for (auto& state : request.query_state()) state.reset();

    // ---- Prefill ----
    for (const auto& input : compiled.inputs()) {
        request.set_tensor(input, make_input_tensor(input, prefill_len, 0));
    }
    auto p0 = std::chrono::steady_clock::now();
    request.infer();
    auto p1 = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(p1 - p0).count();

    InferResult result;
    {
        auto out = request.get_output_tensor();
        result.shape = out.get_shape();
        const float* d = out.data<float>();
        result.logits.assign(d, d + out.get_size());
    }
    std::cout << "  Prefill (seq_len=" << prefill_len << "): "
              << std::fixed << std::setprecision(1) << prefill_ms << " ms" << std::endl;

    if (gen_len <= 0) {
        result.time_ms = prefill_ms;
        return result;
    }

    // ---- Decode loop (no state.reset) ----
    std::cout << "  Decoding " << gen_len << " tokens..." << std::flush;
    double dec_first_ms = 0.0;
    double dec_min_ms = 1e18;
    double dec_max_ms = 0.0;
    double dec_sum_ms = 0.0;
    double dec_warm_sum = 0.0;
    for (int k = 0; k < gen_len; ++k) {
        int past_len = prefill_len + k;
        for (const auto& input : compiled.inputs()) {
            request.set_tensor(input, make_input_tensor(input, 1, past_len));
        }
        auto d0 = std::chrono::steady_clock::now();
        request.infer();
        auto d1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(d1 - d0).count();
        dec_sum_ms += ms;
        if (k == 0) {
            dec_first_ms = ms;
        } else {
            dec_min_ms = std::min(dec_min_ms, ms);
            dec_max_ms = std::max(dec_max_ms, ms);
            dec_warm_sum += ms;
        }
    }
    std::cout << " done" << std::endl;

    int warm_iters = std::max(0, gen_len - 1);
    double dec_avg_warm = warm_iters > 0 ? dec_warm_sum / warm_iters : dec_first_ms;
    double tps = warm_iters > 0 ? 1000.0 * warm_iters / dec_warm_sum : 1000.0 / dec_first_ms;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "    decode first = " << dec_first_ms << " ms" << std::endl;
    if (warm_iters > 0) {
        std::cout << "    decode warm  = avg " << dec_avg_warm
                  << "  min " << dec_min_ms
                  << "  max " << dec_max_ms
                  << " ms  (" << warm_iters << " iters)" << std::endl;
    }
    std::cout << "    decode TPS   = " << std::setprecision(1) << tps << " tok/s" << std::endl;
    std::cout << "    first-token  = " << std::setprecision(2)
              << prefill_ms + dec_first_ms << " ms (prefill + 1st decode)" << std::endl;

    // Use warm decode avg as the headline perf number (this is what TPS reflects).
    result.time_ms = dec_avg_warm;
    return result;
}

static void compare_results(const InferResult& gpu, const InferResult& tp) {
    std::cout << "\n=== Comparison ===" << std::endl;
    std::cout << "  GPU output shape: [";
    for (size_t i = 0; i < gpu.shape.size(); ++i)
        std::cout << (i ? "," : "") << gpu.shape[i];
    std::cout << "]" << std::endl;

    std::cout << "  TP  output shape: [";
    for (size_t i = 0; i < tp.shape.size(); ++i)
        std::cout << (i ? "," : "") << tp.shape[i];
    std::cout << "]" << std::endl;

    if (gpu.shape != tp.shape) {
        std::cerr << "  SHAPE MISMATCH!" << std::endl;
        return;
    }

    size_t total = gpu.logits.size();
    double max_abs_diff = 0.0;
    size_t max_abs_idx = 0;
    double sum_abs_diff = 0.0;
    double sum_sq_diff = 0.0;
    size_t mismatches_1pct = 0;

    for (size_t i = 0; i < total; ++i) {
        double diff = std::abs(static_cast<double>(gpu.logits[i]) - static_cast<double>(tp.logits[i]));
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_abs_idx = i;
        }
        sum_abs_diff += diff;
        sum_sq_diff += diff * diff;

        double denom = std::max(std::abs(static_cast<double>(gpu.logits[i])), 1e-6);
        if (diff / denom > 0.01)
            mismatches_1pct++;
    }

    double mean_abs_diff = sum_abs_diff / static_cast<double>(total);
    double rmse = std::sqrt(sum_sq_diff / static_cast<double>(total));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Max abs diff:  " << max_abs_diff << std::endl;
    {
        size_t vocab = gpu.shape.back();
        size_t seq = gpu.shape.size() >= 2 ? gpu.shape[gpu.shape.size()-2] : 1;
        size_t tok_pos = max_abs_idx / vocab;
        size_t vocab_idx = max_abs_idx % vocab;
        std::cout << "  Max at: flat=" << max_abs_idx << " token=" << tok_pos << " vocab=" << vocab_idx
                  << "  GPU=" << gpu.logits[max_abs_idx] << " TP=" << tp.logits[max_abs_idx] << std::endl;
        // Per-token max error summary
        std::cout << "  Per-token max error (first 8 and last):";
        for (size_t t = 0; t < seq; ++t) {
            double tok_max = 0.0;
            for (size_t v = 0; v < vocab; ++v) {
                double d = std::abs(static_cast<double>(gpu.logits[t*vocab+v]) - static_cast<double>(tp.logits[t*vocab+v]));
                tok_max = std::max(tok_max, d);
            }
            if (t < 8 || t >= seq - 1)
                std::cout << " t" << t << "=" << std::setprecision(3) << tok_max;
            else if (t == 8)
                std::cout << " ...";
        }
        std::cout << std::endl;
    }
    std::cout << "  Mean abs diff: " << mean_abs_diff << std::endl;
    std::cout << "  RMSE:          " << rmse << std::endl;
    std::cout << "  >1% rel diff:  " << mismatches_1pct << " / " << total
              << " (" << std::setprecision(2)
              << 100.0 * mismatches_1pct / total << "%)" << std::endl;

    // Top-5 tokens comparison
    auto top5 = [](const std::vector<float>& logits, size_t vocab_offset, size_t vocab_size) {
        std::vector<size_t> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                          [&](size_t a, size_t b) {
                              return logits[vocab_offset + a] > logits[vocab_offset + b];
                          });
        indices.resize(5);
        return indices;
    };

    // Last token logits
    size_t vocab = gpu.shape.back();
    size_t last_offset = gpu.logits.size() - vocab;

    auto gpu_top5 = top5(gpu.logits, last_offset, vocab);
    auto tp_top5 = top5(tp.logits, last_offset, vocab);

    std::cout << "\n  GPU top-5 tokens: [";
    for (size_t i = 0; i < 5; ++i)
        std::cout << (i ? ", " : "") << gpu_top5[i];
    std::cout << "]" << std::endl;

    std::cout << "  TP  top-5 tokens: [";
    for (size_t i = 0; i < 5; ++i)
        std::cout << (i ? ", " : "") << tp_top5[i];
    std::cout << "]" << std::endl;

    bool top5_match = true;
    for (size_t i = 0; i < 5; ++i) {
        top5_match &= (gpu_top5[i] == tp_top5[i]);
    }
    std::cout << "\n  Top-5 match: " << (top5_match ? "YES" : "NO") << std::endl;

    std::cout << "\n  GPU infer time: " << std::setprecision(1) << gpu.time_ms << " ms" << std::endl;
    std::cout << "  TP  infer time: " << tp.time_ms << " ms" << std::endl;

    // For int4 quantized models, max abs diff up to ~2.0 is normal due to
    // different computation ordering across TP stages.
    if (max_abs_diff < 2.0 && top5_match) {
        std::cout << "\n  RESULT: PASS" << std::endl;
    } else {
        std::cout << "\n  RESULT: FAIL (diff too large or top-5 mismatch)" << std::endl;
    }
}

// Parse --key value or --key=value from argv. Returns empty string if not found.
static std::string parse_arg(int argc, char* argv[], const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == key && i + 1 < argc) return argv[i + 1];
        if (arg.rfind(key + "=", 0) == 0) return arg.substr(key.size() + 1);
    }
    return "";
}

static bool has_flag(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.xml> [tp_degree=2] [seq_len=4]" << std::endl;
        std::cerr << "  --device GPU.X        Reference single-GPU device (default: GPU.0)" << std::endl;
        std::cerr << "  --tp-devices X,Y      Comma-separated TP rank devices (default: GPU.0,GPU.1,...)" << std::endl;
        std::cerr << "  --f32                  Force f32 inference precision" << std::endl;
        std::cerr << "  --no-dq               Disable dynamic quantization (both sides)" << std::endl;
        std::cerr << "  --tp-no-dq            Disable DQ only on TP side (ref keeps DQ)" << std::endl;
        std::cerr << "  --dq-gs N             Set DQ group size (0=disable, default=per-token)" << std::endl;
        std::cerr << "  --no-onednn           Disable oneDNN (use OCL kernels)" << std::endl;
        std::cerr << "  --prefill-len N       Prefill seq_len (default: positional seq_len)" << std::endl;
        std::cerr << "  --gen-len M           Run M decode steps (seq_len=1) after prefill, no reset" << std::endl;
        std::cerr << "  Env: OV_TP_DUMP_AR=1  Dump per-AllReduce rank divergence to stderr" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    uint32_t tp_degree = (argc > 2 && argv[2][0] != '-') ? static_cast<uint32_t>(std::atoi(argv[2])) : 2;
    int seq_len = (argc > 3 && argv[3][0] != '-') ? std::atoi(argv[3]) : 4;

    // Legacy positional arg for backward compat
    bool same_gpu = (argc > 4 && argv[4][0] != '-') && std::string(argv[4]) == "same";
    bool use_f32 = has_flag(argc, argv, "--f32") ||
                   ((argc > 4 && argv[4][0] != '-') && std::string(argv[4]) == "f32");
    bool no_dq = has_flag(argc, argv, "--no-dq");
    bool tp_no_dq = has_flag(argc, argv, "--tp-no-dq");
    bool no_onednn = has_flag(argc, argv, "--no-onednn");
    std::string dq_gs_str = parse_arg(argc, argv, "--dq-gs");
    std::string iters_str = parse_arg(argc, argv, "--iters");
    int iters = iters_str.empty() ? 1 : std::max(1, std::atoi(iters_str.c_str()));
    std::string prefill_len_str = parse_arg(argc, argv, "--prefill-len");
    std::string gen_len_str = parse_arg(argc, argv, "--gen-len");
    int gen_len = gen_len_str.empty() ? 0 : std::max(0, std::atoi(gen_len_str.c_str()));
    int prefill_len = prefill_len_str.empty() ? seq_len : std::max(1, std::atoi(prefill_len_str.c_str()));
    const bool pd_mode = (gen_len > 0);
    // 0 = disable DQ, >0 = group size, empty = don't set (use default per-token)
    int64_t dq_gs = dq_gs_str.empty() ? -1 : std::stoll(dq_gs_str);

    // New named args
    std::string ref_device = parse_arg(argc, argv, "--device");
    if (ref_device.empty()) ref_device = "GPU.0";

    std::string tp_devices_str = parse_arg(argc, argv, "--tp-devices");
    std::vector<std::string> tp_device_list;
    if (!tp_devices_str.empty()) {
        // Parse comma-separated list
        std::string token;
        for (size_t i = 0; i <= tp_devices_str.size(); ++i) {
            if (i == tp_devices_str.size() || tp_devices_str[i] == ',') {
                if (!token.empty()) tp_device_list.push_back(token);
                token.clear();
            } else {
                token += tp_devices_str[i];
            }
        }
    }

    std::cout << "Model:     " << model_path << std::endl;
    std::cout << "TP degree: " << tp_degree << std::endl;
    if (pd_mode) {
        std::cout << "Mode:      prefill+decode  prefill_len=" << prefill_len
                  << "  gen_len=" << gen_len << std::endl;
    } else {
        std::cout << "Seq len:   " << seq_len << std::endl;
    }
    std::cout << "Ref device:" << ref_device << std::endl;
    if (!tp_device_list.empty()) {
        std::cout << "TP devices:";
        for (const auto& d : tp_device_list) std::cout << " " << d;
        std::cout << std::endl;
    }
    if (same_gpu)
        std::cout << "Mode:      SAME GPU (both ranks on " << ref_device << ")" << std::endl;
    std::cout << std::endl;

    try {
        ov::Core core;
        std::cout << "Available devices: ";
        for (const auto& d : core.get_available_devices())
            std::cout << d << " ";
        std::cout << std::endl;

        // Register TP plugin if not already auto-registered
        try {
            core.register_plugin("openvino_tensor_parallel_plugin", "TENSOR_PARALLEL");
            std::cout << "Registered TENSOR_PARALLEL plugin" << std::endl << std::endl;
        } catch (const std::exception&) {
            std::cout << "TENSOR_PARALLEL plugin already registered" << std::endl << std::endl;
        }

        auto model = core.read_model(model_path);
        std::cout << "Model loaded: " << model->get_ordered_ops().size() << " ops" << std::endl;
        std::cout << std::endl;

        // --- Run on single GPU ---
        std::cout << "[1] Single GPU (" << ref_device << "):" << std::endl;
        ov::AnyMap gpu_config = {};
        if (use_f32) {
            gpu_config[ov::hint::inference_precision.name()] = ov::element::f32;
            std::cout << "  (forcing f32 inference precision)" << std::endl;
        }
        if (no_dq) {
            gpu_config[ov::hint::dynamic_quantization_group_size.name()] = (uint64_t)0;
            std::cout << "  (disabling dynamic quantization)" << std::endl;
        } else if (dq_gs >= 0) {
            gpu_config[ov::hint::dynamic_quantization_group_size.name()] = (uint64_t)dq_gs;
            std::cout << "  (DQ group_size=" << dq_gs << ")" << std::endl;
        } else {
            // Match the TP plugin's default: disable DQ for fair comparison.
            // DQ + oneDNN FC produces shape-dependent results due to BRGEMM tiling,
            // causing >1.0 max_abs_diff at seq_len >= 80 in TP mode.
            gpu_config[ov::hint::dynamic_quantization_group_size.name()] = (uint64_t)0;
            std::cout << "  (DQ disabled, matching TP default)" << std::endl;
        }
        if (no_onednn) {
            // GPU_USE_ONEDNN is a RELEASE_INTERNAL property — can't set via public API.
            // Use env var which gets picked up by apply_env_options() during finalize.
            // Requires ENABLE_DEBUG_CAPS build.
            setenv("OV_GPU_USE_ONEDNN", "0", 1);
            std::cout << "  (disabling oneDNN via OV_GPU_USE_ONEDNN=0)" << std::endl;
        }
        auto gpu_result = pd_mode
            ? run_prefill_decode(core, model, ref_device, gpu_config, prefill_len, gen_len)
            : run_inference(core, model, ref_device, gpu_config, seq_len, iters);

        // --- Run on TENSOR_PARALLEL ---
        std::cout << "\n[2] Tensor Parallel (degree=" << tp_degree << "):" << std::endl;
        ov::AnyMap tp_config = {
            {"TENSOR_PARALLEL_DEGREE", tp_degree},
        };

        if (!tp_device_list.empty()) {
            tp_config["TENSOR_PARALLEL_DEVICES"] = tp_device_list;
        } else if (same_gpu) {
            tp_config["TENSOR_PARALLEL_DEVICES"] = std::vector<std::string>{ref_device, ref_device};
        }

        if (use_f32) {
            tp_config[ov::hint::inference_precision.name()] = ov::element::f32;
            std::cout << "  (forcing f32 inference precision)" << std::endl;
        }
        if (no_dq || tp_no_dq) {
            tp_config[ov::hint::dynamic_quantization_group_size.name()] = (uint64_t)0;
            std::cout << "  (disabling dynamic quantization" << (tp_no_dq ? " on TP side only" : "") << ")" << std::endl;
        } else if (dq_gs >= 0) {
            tp_config[ov::hint::dynamic_quantization_group_size.name()] = (uint64_t)dq_gs;
            std::cout << "  (DQ group_size=" << dq_gs << ")" << std::endl;
        }
        if (no_onednn) {
            std::cout << "  (disabling oneDNN via OV_GPU_USE_ONEDNN=0)" << std::endl;
        }

        auto tp_result = pd_mode
            ? run_prefill_decode(core, model, "TENSOR_PARALLEL", tp_config, prefill_len, gen_len)
            : run_inference(core, model, "TENSOR_PARALLEL", tp_config, seq_len, iters);

        // --- Compare ---
        compare_results(gpu_result, tp_result);

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
