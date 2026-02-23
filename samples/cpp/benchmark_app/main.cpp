// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "openvino/pass/serialize.hpp"

#ifndef IN_OV_COMPONENT
#    define IN_OV_COMPONENT
#    define WAS_OV_LIBRARY_DEFINED
#endif

// #include "openvino/pass/extract_first_sdpa.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

#ifdef WAS_OV_LIBRARY_DEFINED
#    undef IN_OV_COMPONENT
#    undef WAS_OV_LIBRARY_DEFINED
#endif

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"

#include "benchmark_app.hpp"
#include "infer_request_wrap.hpp"
#include "inputs_filling.hpp"
#include "remote_tensors_filling.hpp"
#include "statistics_report.hpp"
#include "utils.hpp"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <sys/resource.h>
#elif defined(__linux__)
#include <regex>
#include <sstream>
#else
#error "unsupported OS"
#endif

enum class ModelType {
    INPUT_IDS,      // Standard LLM (e.g. Phi-4): uses input_ids (i64)
    INPUTS_EMBEDS   // Vision-Language (e.g. Qwen2.5-VL): uses inputs_embeds (f32)
};

// Print the last written slot in the KV cache
// Cache layout: [num_blocks, num_kv_heads, block_size, head_dim]
void print_cache_slot(const std::string& name, const ov::Tensor& cache, int64_t seq_len) {
    auto shape = cache.get_shape();
    auto* data = cache.data<float>();
    size_t num_kv_heads = shape[1];
    size_t block_size = shape[2];
    size_t head_dim = shape[3];

    // Last token is at zero-based index (seq_len - 1).
    // block = which block it falls into, slot = position within that block.
    // E.g. seq_len=260, block_size=32: token 259 → block 8, slot 3.
    size_t block = (seq_len - 1) / block_size;
    size_t slot = (seq_len - 1) % block_size;

    std::cout << name << " shape=[" << shape[0] << "," << shape[1] << "," << shape[2] << "," << shape[3]
              << "] last token at block=" << block << " slot=" << slot << " head0 first 4 vals: ";
    size_t offset = block * (num_kv_heads * block_size * head_dim) + 0 * (block_size * head_dim) + slot * head_dim;
    for (int i = 0; i < 4; i++) {
        std::cout << std::fixed << std::setprecision(4) << data[offset + i] << " ";
    }
    std::cout << std::endl;
}

// Helper to print tensor shape
void print_shape(const std::string& name, const ov::Shape& shape) {
    std::cout << name << ": [";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
}

// Compare outputs with layout mapping (assumes batch=1)
// SDPA shape: [batch, heads, seq, dim] — head-major layout
// PA shape:   [tokens, heads, 1, dim] — token-major layout
// Index formulas skip the batch dimension, so this only works with batch=1.
void compare_outputs(const float* sdpa_data, const ov::Shape& sdpa_shape,
                     const float* pa_data, const ov::Shape& pa_shape,
                     const std::string& phase) {
    const size_t num_heads = sdpa_shape[1];
    const size_t seq_len = sdpa_shape[2];
    const size_t head_dim = sdpa_shape[3];

    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    size_t count = 0;

    for (size_t s = 0; s < seq_len; s++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t d = 0; d < head_dim; d++) {
                // SDPA: [batch=1, heads, seq, dim]
                size_t sdpa_idx = h * seq_len * head_dim + s * head_dim + d;
                // PA: [tokens, heads, 1, dim]
                size_t pa_idx = s * num_heads * head_dim + h * head_dim + d;

                float diff = std::abs(sdpa_data[sdpa_idx] - pa_data[pa_idx]);
                max_diff = std::max(max_diff, diff);
                sum_diff += diff;
                count++;
            }
        }
    }

    std::cout << phase << ": Max diff: " << std::scientific << std::setprecision(4) << max_diff
              << ", Mean diff: " << (sum_diff / count) << std::fixed << std::endl;
}

// Compare logits (assumes batch=1): SDPA [1, seq, vocab] vs PA [seq, 1, vocab].
// The "1" in different positions doesn't affect strides, so linear memory layout
// is identical and we can do direct element-by-element comparison.
void compare_logits(const float* sdpa_data, const ov::Shape& sdpa_shape,
                    const float* pa_data, const ov::Shape& pa_shape,
                    const std::string& phase) {
    // SDPA: [1, seq, vocab], PA: [seq, vocab] — batch=1 so data layout is identical
    size_t total = 1;
    for (auto d : pa_shape) total *= d;

    float max_diff = 0.0f;
    double sum_diff = 0.0;

    for (size_t i = 0; i < total; i++) {
        float diff = std::abs(sdpa_data[i] - pa_data[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
    }

    std::cout << phase << ": Max diff: " << std::scientific << std::setprecision(4) << max_diff
              << ", Mean diff: " << (sum_diff / total) << std::fixed << std::endl;

    // Print first 15 logit values from the last token
    size_t last_token_offset = total - pa_shape.back();  // offset to last token's logits
    std::cout << "  SDPA logits[0:15]: ";
    for (size_t i = 0; i < 15; i++)
        std::cout << std::fixed << std::setprecision(4) << sdpa_data[last_token_offset + i] << " ";
    std::cout << std::endl;
    std::cout << "  PA   logits[0:15]: ";
    for (size_t i = 0; i < 15; i++)
        std::cout << std::fixed << std::setprecision(4) << pa_data[last_token_offset + i] << " ";
    std::cout << std::endl;
}

/**
 * @brief The entry point of the benchmark application
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_dir> [FULL_MODEL]" << std::endl;
        std::cout << "  model_dir should contain openvino_model.xml/.bin and config.json" << std::endl;
        std::cout << "  FULL_MODEL - run all layers instead of extracting first SDPA" << std::endl;
        return 1;
    }

    std::string model_dir = argv[1];
    bool full_model = (argc >= 3 && std::string(argv[2]) == "FULL_MODEL");
    std::string config_path = model_dir + "/config.json";

    // Try openvino_model.xml first, fall back to openvino_language_model.xml
    std::string model_path = model_dir + "/openvino_model.xml";
    if (!std::ifstream(model_path).good()) {
        model_path = model_dir + "/openvino_language_model.xml";
    }

    // Read model config
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "Failed to open " << config_path << std::endl;
        return 1;
    }
    nlohmann::json config = nlohmann::json::parse(config_file);

    const int64_t num_kv_heads = config["num_key_value_heads"];
    const int64_t num_q_heads = config["num_attention_heads"];
    const int64_t hidden_size = config["hidden_size"];
    const int64_t head_dim = hidden_size / num_q_heads;
    const int64_t num_hidden_layers = config.value("num_hidden_layers", 1);
    const int64_t num_layers = full_model ? num_hidden_layers : 1;

    // Detect model type from config
    std::string model_type_str = config.value("model_type", "");
    ModelType model_type = ModelType::INPUT_IDS;
    if (model_type_str == "qwen2_5_vl") {
        model_type = ModelType::INPUTS_EMBEDS;
    }

    std::cout << "Mode: " << (full_model ? "FULL_MODEL" : "SINGLE_LAYER") << std::endl;
    std::cout << "Model config: num_kv_heads=" << num_kv_heads
              << ", num_q_heads=" << num_q_heads
              << ", head_dim=" << head_dim
              << ", num_layers=" << num_layers
              << ", model_type=" << model_type_str
              << " (" << (model_type == ModelType::INPUTS_EMBEDS ? "INPUTS_EMBEDS" : "INPUT_IDS") << ")"
              << std::endl;

    ov::Core core;

    // Common test parameters
    const int64_t batch = 1;
    const int64_t block_size = 32;    // PA block size (must be 32 for CPU)

    // Model-specific test configuration
    int64_t prefill_len;
    int64_t num_decode_iterations = 11;

    // M-RoPE position arrays (only used for INPUTS_EMBEDS / qwen2_5_vl)
    std::vector<int64_t> mrope_temporal, mrope_height, mrope_width;

    // For INPUT_IDS models: token array
    std::vector<int64_t> all_tokens;

    // For INPUTS_EMBEDS models: embedding array
    std::vector<float> all_embeddings;

    // RNG for synthetic embeddings
    std::mt19937 rng(42);
    std::normal_distribution<float> embed_dist(0.0f, 0.02f);

    if (model_type == ModelType::INPUT_IDS) {
        prefill_len = 4;
        int64_t total_len = prefill_len + num_decode_iterations;
        for (int64_t i = 0; i < total_len; i++) {
            all_tokens.push_back(100 + i * 111);
        }
    } else {
        // Qwen2.5-VL style: simulate 448x448 image
        // patch_size=14, spatial_merge_size=2 -> effective merge=28 -> grid=448/28=16x16=256 image tokens
        // Layout: 2 text + 256 image + 2 text = 260 prefill tokens
        const int64_t num_text_before = 2;
        const int64_t grid_h = 16, grid_w = 16;
        const int64_t num_image_tokens = grid_h * grid_w;  // 256
        const int64_t num_text_after = 2;
        prefill_len = num_text_before + num_image_tokens + num_text_after;  // 260

        int64_t total_len = prefill_len + num_decode_iterations;

        // Generate synthetic embeddings for all tokens
        all_embeddings.resize(total_len * hidden_size);
        for (auto& v : all_embeddings)
            v = embed_dist(rng);

        // Build M-RoPE position arrays for prefill
        // Text before image: all 3 planes same
        for (int64_t i = 0; i < num_text_before; i++) {
            mrope_temporal.push_back(i);
            mrope_height.push_back(i);
            mrope_width.push_back(i);
        }
        // Image tokens: 16x16 grid
        int64_t img_start_pos = num_text_before;  // = 2
        for (int64_t row = 0; row < grid_h; row++) {
            for (int64_t col = 0; col < grid_w; col++) {
                mrope_temporal.push_back(img_start_pos);            // same for all image tokens
                mrope_height.push_back(img_start_pos + row);        // varies by row: 2..17
                mrope_width.push_back(img_start_pos + col);         // varies by col: 2..17
            }
        }
        // Text after image: all 3 planes same, continue from max image position + 1
        int64_t after_img_pos = img_start_pos + grid_h;  // 2 + 16 = 18
        for (int64_t i = 0; i < num_text_after; i++) {
            mrope_temporal.push_back(after_img_pos + i);
            mrope_height.push_back(after_img_pos + i);
            mrope_width.push_back(after_img_pos + i);
        }

        std::cout << "Image simulation: " << num_text_before << " text + "
                  << num_image_tokens << " image (" << grid_h << "x" << grid_w
                  << ") + " << num_text_after << " text = " << prefill_len << " prefill tokens" << std::endl;
    }

    int64_t total_len = prefill_len + num_decode_iterations;
    int64_t num_blocks = (prefill_len + block_size - 1) / block_size;  // start small, grow dynamically

    std::cout << "Test Configuration" << std::endl;
    std::cout << "Prefill tokens: " << prefill_len << std::endl;
    std::cout << "Decode iterations: " << num_decode_iterations << std::endl;
    std::cout << "Total sequence length: " << total_len << std::endl;
    std::cout << "Initial blocks allocated: " << num_blocks << std::endl;
    std::cout << std::endl;

    // --- Lambdas for setting primary input (model-type aware) ---

    auto set_sdpa_primary_input_prefill = [&](ov::InferRequest& req) {
        if (model_type == ModelType::INPUT_IDS) {
            auto input_ids = ov::Tensor(ov::element::i64, {(size_t)batch, (size_t)prefill_len});
            for (int64_t i = 0; i < prefill_len; i++)
                input_ids.data<int64_t>()[i] = all_tokens[i];
            req.set_tensor("input_ids", input_ids);
        } else {
            auto embeds = ov::Tensor(ov::element::f32, {(size_t)batch, (size_t)prefill_len, (size_t)hidden_size});
            std::memcpy(embeds.data<float>(), all_embeddings.data(),
                        prefill_len * hidden_size * sizeof(float));
            req.set_tensor("inputs_embeds", embeds);
        }
    };

    auto set_pa_primary_input_prefill = [&](ov::InferRequest& req) {
        if (model_type == ModelType::INPUT_IDS) {
            auto input_ids = ov::Tensor(ov::element::i64, {(size_t)prefill_len});
            for (int64_t i = 0; i < prefill_len; i++)
                input_ids.data<int64_t>()[i] = all_tokens[i];
            req.set_tensor("input_ids", input_ids);
        } else {
            auto embeds = ov::Tensor(ov::element::f32, {(size_t)prefill_len, (size_t)hidden_size});
            std::memcpy(embeds.data<float>(), all_embeddings.data(),
                        prefill_len * hidden_size * sizeof(float));
            req.set_tensor("inputs_embeds", embeds);
        }
    };

    auto set_sdpa_primary_input_decode = [&](ov::InferRequest& req, int64_t iter) {
        if (model_type == ModelType::INPUT_IDS) {
            auto input_ids = ov::Tensor(ov::element::i64, {(size_t)batch, 1});
            input_ids.data<int64_t>()[0] = all_tokens[prefill_len + iter];
            req.set_tensor("input_ids", input_ids);
        } else {
            auto embeds = ov::Tensor(ov::element::f32, {(size_t)batch, 1, (size_t)hidden_size});
            std::memcpy(embeds.data<float>(),
                        all_embeddings.data() + (prefill_len + iter) * hidden_size,
                        hidden_size * sizeof(float));
            req.set_tensor("inputs_embeds", embeds);
        }
    };

    auto set_pa_primary_input_decode = [&](ov::InferRequest& req, int64_t iter) {
        if (model_type == ModelType::INPUT_IDS) {
            auto input_ids = ov::Tensor(ov::element::i64, {1});
            input_ids.data<int64_t>()[0] = all_tokens[prefill_len + iter];
            req.set_tensor("input_ids", input_ids);
        } else {
            auto embeds = ov::Tensor(ov::element::f32, {1, (size_t)hidden_size});
            std::memcpy(embeds.data<float>(),
                        all_embeddings.data() + (prefill_len + iter) * hidden_size,
                        hidden_size * sizeof(float));
            req.set_tensor("inputs_embeds", embeds);
        }
    };

    // --- Lambdas for setting position_ids (model-type aware) ---

    auto set_sdpa_position_ids_prefill = [&](ov::InferRequest& req) {
        if (model_type == ModelType::INPUT_IDS) {
            auto pos_ids = ov::Tensor(ov::element::i64, {(size_t)batch, (size_t)prefill_len});
            for (int64_t i = 0; i < prefill_len; i++)
                pos_ids.data<int64_t>()[i] = i;
            req.set_tensor("position_ids", pos_ids);
        } else {
            // M-RoPE: [3, batch, seq_len]
            auto pos_ids = ov::Tensor(ov::element::i64, {3, (size_t)batch, (size_t)prefill_len});
            auto* data = pos_ids.data<int64_t>();
            for (int64_t i = 0; i < prefill_len; i++)
                data[0 * prefill_len + i] = mrope_temporal[i];
            for (int64_t i = 0; i < prefill_len; i++)
                data[1 * prefill_len + i] = mrope_height[i];
            for (int64_t i = 0; i < prefill_len; i++)
                data[2 * prefill_len + i] = mrope_width[i];
            req.set_tensor("position_ids", pos_ids);
        }
    };

    auto set_pa_position_ids_prefill = [&](ov::InferRequest& req) {
        if (model_type == ModelType::INPUT_IDS) {
            auto pos_ids = ov::Tensor(ov::element::i64, {(size_t)prefill_len});
            for (int64_t i = 0; i < prefill_len; i++)
                pos_ids.data<int64_t>()[i] = i;
            req.set_tensor("position_ids", pos_ids);
        } else {
            // M-RoPE: flat [3*seq_len] — transformation reshapes internally
            auto pos_ids = ov::Tensor(ov::element::i64, {3 * (size_t)prefill_len});
            auto* data = pos_ids.data<int64_t>();
            for (int64_t i = 0; i < prefill_len; i++)
                data[i] = mrope_temporal[i];
            for (int64_t i = 0; i < prefill_len; i++)
                data[prefill_len + i] = mrope_height[i];
            for (int64_t i = 0; i < prefill_len; i++)
                data[2 * prefill_len + i] = mrope_width[i];
            req.set_tensor("position_ids", pos_ids);
        }
    };

    auto set_sdpa_position_ids_decode = [&](ov::InferRequest& req, int64_t position) {
        if (model_type == ModelType::INPUT_IDS) {
            auto pos_ids = ov::Tensor(ov::element::i64, {(size_t)batch, 1});
            pos_ids.data<int64_t>()[0] = position;
            req.set_tensor("position_ids", pos_ids);
        } else {
            // M-RoPE: [3, batch, 1] - all planes same for text decode tokens
            auto pos_ids = ov::Tensor(ov::element::i64, {3, (size_t)batch, 1});
            auto* data = pos_ids.data<int64_t>();
            data[0] = position;  // temporal
            data[1] = position;  // height
            data[2] = position;  // width
            req.set_tensor("position_ids", pos_ids);
        }
    };

    auto set_pa_position_ids_decode = [&](ov::InferRequest& req, int64_t position) {
        if (model_type == ModelType::INPUT_IDS) {
            auto pos_ids = ov::Tensor(ov::element::i64, {1});
            pos_ids.data<int64_t>()[0] = position;
            req.set_tensor("position_ids", pos_ids);
        } else {
            // M-RoPE: flat [3] — transformation reshapes internally
            auto pos_ids = ov::Tensor(ov::element::i64, {3});
            auto* data = pos_ids.data<int64_t>();
            data[0] = position;  // temporal
            data[1] = position;  // height
            data[2] = position;  // width
            req.set_tensor("position_ids", pos_ids);
        }
    };

    // --- Read and transform the model ---

    std::cout << "Reading model from " << model_path << std::endl;
    auto model = core.read_model(model_path);

    if (!full_model) {
        // ov::pass::ExtractFirstSDPA().run_on_model(model);
        std::cout << "ExtractFirstSDPA done." << std::endl;
    } else {
        std::cout << "Skipping ExtractFirstSDPA (FULL_MODEL mode)." << std::endl;
    }

    // Clone for SDPA use, then transform original into PA
    auto sdpa_model = model->clone();

    ov::pass::SDPAToPagedAttention().run_on_model(model);
    auto pa_model = model;
    std::cout << "SDPAToPagedAttention done." << std::endl;

    // Compile both models
    auto sdpa_compiled = core.compile_model(sdpa_model, "CPU",
        ov::hint::inference_precision(ov::element::f32),
        ov::key_cache_precision(ov::element::f32),
        ov::value_cache_precision(ov::element::f32));
    auto sdpa_request = sdpa_compiled.create_infer_request();

    auto pa_compiled = core.compile_model(pa_model, "CPU",
        ov::hint::inference_precision(ov::element::f32),
        ov::key_cache_precision(ov::element::f32),
        ov::value_cache_precision(ov::element::f32));
    auto pa_request = pa_compiled.create_infer_request();

    std::cout << "Models compiled successfully." << std::endl << std::endl;

    // --- KV cache tensors for PA ---

    auto make_cache = [&]() {
        auto t = ov::Tensor(ov::element::f32,
            {(size_t)num_blocks, (size_t)num_kv_heads, (size_t)block_size, (size_t)head_dim});
        std::fill_n(t.data<float>(), t.get_size(), 0.0f);
        return t;
    };

    std::vector<ov::Tensor> key_caches(num_layers), value_caches(num_layers);
    for (int64_t l = 0; l < num_layers; l++) {
        key_caches[l] = make_cache();
        value_caches[l] = make_cache();
    }

    // Helper to grow all cache tensors when more blocks are needed
    auto grow_cache_if_needed = [&](int64_t required_seq_len) {
        int64_t required_blocks = (required_seq_len + block_size - 1) / block_size;
        if (required_blocks <= num_blocks) return;

        int64_t old_blocks = num_blocks;
        num_blocks = required_blocks;
        std::cout << "  ** Growing cache: " << old_blocks << " -> " << num_blocks << " blocks **" << std::endl;

        for (int64_t l = 0; l < num_layers; l++) {
            auto new_kc = ov::Tensor(ov::element::f32,
                {(size_t)num_blocks, (size_t)num_kv_heads, (size_t)block_size, (size_t)head_dim});
            std::fill_n(new_kc.data<float>(), new_kc.get_size(), 0.0f);
            std::memcpy(new_kc.data<float>(), key_caches[l].data<float>(), key_caches[l].get_byte_size());
            key_caches[l] = new_kc;

            auto new_vc = ov::Tensor(ov::element::f32,
                {(size_t)num_blocks, (size_t)num_kv_heads, (size_t)block_size, (size_t)head_dim});
            std::fill_n(new_vc.data<float>(), new_vc.get_size(), 0.0f);
            std::memcpy(new_vc.data<float>(), value_caches[l].data<float>(), value_caches[l].get_byte_size());
            value_caches[l] = new_vc;
        }
    };

    // Helper to set all cache tensors on PA request
    auto set_pa_caches = [&](ov::InferRequest& req) {
        for (int64_t l = 0; l < num_layers; l++) {
            req.set_tensor("key_cache." + std::to_string(l), key_caches[l]);
            req.set_tensor("value_cache." + std::to_string(l), value_caches[l]);
        }
    };

    // Compute decode starting position for M-RoPE
    int64_t mrope_next_position = 0;
    if (model_type == ModelType::INPUTS_EMBEDS && !mrope_temporal.empty()) {
        mrope_next_position = std::max({mrope_temporal.back(), mrope_height.back(), mrope_width.back()}) + 1;
    }

    // ===== PREFILL Phase =====
    std::cout << "PREFILL Phase (tokens: " << prefill_len << ")" << std::endl;

    // SDPA Prefill
    {
        sdpa_request.reset_state();

        set_sdpa_primary_input_prefill(sdpa_request);
        set_sdpa_position_ids_prefill(sdpa_request);

        // attention_mask: [batch, prefill_len]
        auto attn_mask = ov::Tensor(ov::element::i64, {(size_t)batch, (size_t)prefill_len});
        std::fill_n(attn_mask.data<int64_t>(), prefill_len, 1);
        sdpa_request.set_tensor("attention_mask", attn_mask);

        // beam_idx: [batch]
        auto beam_idx = ov::Tensor(ov::element::i32, {(size_t)batch});
        beam_idx.data<int32_t>()[0] = 0;
        sdpa_request.set_tensor("beam_idx", beam_idx);

        sdpa_request.infer();
        print_shape(full_model ? "SDPA prefill logits" : "SDPA prefill output", sdpa_request.get_output_tensor(0).get_shape());
        auto states = sdpa_request.query_state();
        if (full_model) {
            std::cout << "  SDPA states: " << states.size() << " total" << std::endl;
        } else {
            for (auto& state : states) {
                auto shape = state.get_state().get_shape();
                std::cout << "  SDPA state '" << state.get_name() << "' shape=[";
                for (size_t i = 0; i < shape.size(); i++) std::cout << (i ? "," : "") << shape[i];
                std::cout << "]" << std::endl;
            }
        }
    }

    // PA Prefill
    {
        set_pa_primary_input_prefill(pa_request);
        set_pa_position_ids_prefill(pa_request);

        // Set caches (PA writes to these in-place)
        set_pa_caches(pa_request);

        // past_lens: [batch] - 0 for prefill
        auto past_lens = ov::Tensor(ov::element::i32, {(size_t)batch});
        past_lens.data<int32_t>()[0] = 0;
        pa_request.set_tensor("past_lens", past_lens);

        // subsequence_begins: [batch + 1]
        auto subseq_begins = ov::Tensor(ov::element::i32, {(size_t)(batch + 1)});
        subseq_begins.data<int32_t>()[0] = 0;
        subseq_begins.data<int32_t>()[1] = prefill_len;
        pa_request.set_tensor("subsequence_begins", subseq_begins);

        // block_indices: [total_blocks_used]
        auto block_indices = ov::Tensor(ov::element::i32, {(size_t)num_blocks});
        for (int32_t i = 0; i < num_blocks; i++)
            block_indices.data<int32_t>()[i] = i;
        pa_request.set_tensor("block_indices", block_indices);

        // block_indices_begins: [batch + 1]
        auto block_idx_begins = ov::Tensor(ov::element::i32, {(size_t)(batch + 1)});
        block_idx_begins.data<int32_t>()[0] = 0;
        block_idx_begins.data<int32_t>()[1] = num_blocks;
        pa_request.set_tensor("block_indices_begins", block_idx_begins);

        // max_context_len: scalar
        auto max_ctx_len = ov::Tensor(ov::element::i32, {});
        max_ctx_len.data<int32_t>()[0] = prefill_len;
        pa_request.set_tensor("max_context_len", max_ctx_len);

        pa_request.infer();
        print_shape(full_model ? "PA prefill logits" : "PA prefill output", pa_request.get_output_tensor(0).get_shape());
        // print_cache_slot("  key_cache.0 after prefill", key_caches[0], prefill_len);
        // print_cache_slot("  val_cache.0 after prefill", value_caches[0], prefill_len);
    }

    // Compare prefill outputs
    {
        auto sdpa_out = sdpa_request.get_output_tensor(0);
        auto pa_out = pa_request.get_output_tensor(0);
        if (full_model) {
            compare_logits(sdpa_out.data<float>(), sdpa_out.get_shape(),
                           pa_out.data<float>(), pa_out.get_shape(),
                           "PREFILL");
        } else {
            compare_outputs(sdpa_out.data<float>(), sdpa_out.get_shape(),
                            pa_out.data<float>(), pa_out.get_shape(),
                            "PREFILL");
        }
    }

    // ===== DECODE Phase =====
    int64_t current_seq_len = prefill_len;

    for (int64_t iter = 0; iter < num_decode_iterations; iter++) {
        int64_t new_position;
        if (model_type == ModelType::INPUTS_EMBEDS) {
            new_position = mrope_next_position + iter;
        } else {
            new_position = current_seq_len;
        }

        std::cout << "\nDECODE Iteration " << iter + 1
                  << " (pos: " << new_position << ")" << std::endl;

        // SDPA Decode
        {
            set_sdpa_primary_input_decode(sdpa_request, iter);
            set_sdpa_position_ids_decode(sdpa_request, new_position);

            // attention_mask: [batch, current_seq_len + 1] - extend mask
            auto attn_mask = ov::Tensor(ov::element::i64, {(size_t)batch, (size_t)(current_seq_len + 1)});
            std::fill_n(attn_mask.data<int64_t>(), current_seq_len + 1, 1);
            sdpa_request.set_tensor("attention_mask", attn_mask);

            // beam_idx: [batch]
            auto beam_idx = ov::Tensor(ov::element::i32, {(size_t)batch});
            beam_idx.data<int32_t>()[0] = 0;
            sdpa_request.set_tensor("beam_idx", beam_idx);

            sdpa_request.infer();
            print_shape(full_model ? "SDPA decode logits" : "SDPA decode output", sdpa_request.get_output_tensor(0).get_shape());
            if (!full_model) {
                for (auto& state : sdpa_request.query_state()) {
                    auto shape = state.get_state().get_shape();
                    std::cout << "  SDPA state '" << state.get_name() << "' shape=[";
                    for (size_t i = 0; i < shape.size(); i++) std::cout << (i ? "," : "") << shape[i];
                    std::cout << "]" << std::endl;
                }
            }
        }

        // PA Decode
        {
            set_pa_primary_input_decode(pa_request, iter);
            set_pa_position_ids_decode(pa_request, new_position);

            // Grow cache if needed before setting tensors
            grow_cache_if_needed(current_seq_len + 1);

            // Caches - PA will append new KV in-place
            set_pa_caches(pa_request);

            // past_lens: [batch] - number of tokens already in cache
            auto past_lens = ov::Tensor(ov::element::i32, {(size_t)batch});
            past_lens.data<int32_t>()[0] = current_seq_len;
            pa_request.set_tensor("past_lens", past_lens);

            // subsequence_begins: [batch + 1] - for 1 token
            auto subseq_begins = ov::Tensor(ov::element::i32, {(size_t)(batch + 1)});
            subseq_begins.data<int32_t>()[0] = 0;
            subseq_begins.data<int32_t>()[1] = 1;  // 1 new token
            pa_request.set_tensor("subsequence_begins", subseq_begins);

            // block_indices
            auto block_indices = ov::Tensor(ov::element::i32, {(size_t)num_blocks});
            for (int32_t i = 0; i < num_blocks; i++)
                block_indices.data<int32_t>()[i] = i;
            pa_request.set_tensor("block_indices", block_indices);

            // block_indices_begins
            auto block_idx_begins = ov::Tensor(ov::element::i32, {(size_t)(batch + 1)});
            block_idx_begins.data<int32_t>()[0] = 0;
            block_idx_begins.data<int32_t>()[1] = num_blocks;
            pa_request.set_tensor("block_indices_begins", block_idx_begins);

            // max_context_len: current total length
            auto max_ctx_len = ov::Tensor(ov::element::i32, {});
            max_ctx_len.data<int32_t>()[0] = current_seq_len + 1;
            pa_request.set_tensor("max_context_len", max_ctx_len);

            pa_request.infer();
            print_shape(full_model ? "PA decode logits" : "PA decode output", pa_request.get_output_tensor(0).get_shape());
            // print_cache_slot("  key_cache.0 after decode", key_caches[0], current_seq_len + 1);
            // print_cache_slot("  val_cache.0 after decode", value_caches[0], current_seq_len + 1);
        }

        // Compare decode outputs
        {
            auto sdpa_out = sdpa_request.get_output_tensor(0);
            auto pa_out = pa_request.get_output_tensor(0);
            if (full_model) {
                compare_logits(sdpa_out.data<float>(), sdpa_out.get_shape(),
                               pa_out.data<float>(), pa_out.get_shape(),
                               "DECODE " + std::to_string(iter + 1));
            } else {
                compare_outputs(sdpa_out.data<float>(), sdpa_out.get_shape(),
                                pa_out.data<float>(), pa_out.get_shape(),
                                "DECODE " + std::to_string(iter + 1));
            }
        }

        current_seq_len++;
    }

    std::cout << "\nSummary" << std::endl;
    std::cout << "Successfully ran " << (1 + num_decode_iterations) << " iterations" << std::endl;
    std::cout << "Final sequence length: " << current_seq_len << std::endl;

    return 0;
}
