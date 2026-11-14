// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "openvino/openvino.hpp"

ov::Core& ov_singleton_core();

namespace {

struct LogState {
    int offloaded_layers = 0;
    int total_layers = 0;
};

void log_message(ggml_log_level level, const char* message, void* user_data) {
    auto& state = *static_cast<LogState*>(user_data);
    if (const char* offloaded = std::strstr(message, "offloaded ")) {
        int layers = 0;
        int total = 0;
        if (std::sscanf(offloaded, "offloaded %d/%d layers to GPU", &layers, &total) == 2 && total > 0) {
            state.offloaded_layers = layers;
            state.total_layers = total;
        }
    }
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(message, stderr);
    }
}

struct Batch {
    llama_batch value = llama_batch_init(2, 0, 1);
    ~Batch() {
        llama_batch_free(value);
    }
};

void restore_sequences(llama_context* context, int first_seq) {
    std::vector<std::vector<uint8_t>> states(2);
    for (int seq = 0; seq < 2; ++seq) {
        const auto id = first_seq + seq;
        const size_t size = llama_state_seq_get_size(context, id);
        if (size == 0) {
            throw std::runtime_error("empty sequence state");
        }
        states[seq].resize(size);
        if (llama_state_seq_get_data(context, states[seq].data(), size, id) != size) {
            throw std::runtime_error("failed to save sequence state");
        }
    }
    for (int seq = 0; seq < 2; ++seq) {
        if (!llama_memory_seq_rm(llama_get_memory(context), first_seq + seq, -1, -1)) {
            throw std::runtime_error("failed to remove sequence state");
        }
    }
    for (int seq = 0; seq < 2; ++seq) {
        const auto& state = states[seq];
        if (llama_state_seq_set_data(context, state.data(), state.size(), first_seq + seq) != state.size()) {
            throw std::runtime_error("failed to restore sequence state");
        }
    }
}

std::vector<float> run(const char* model_path,
                       ggml_backend_dev_t device,
                       bool flash,
                       int first_seq,
                       bool restore,
                       LogState& logs) {
    const bool offload = device != nullptr;
    auto model_params = llama_model_default_params();
    ggml_backend_dev_t devices[] = {device, nullptr};
    model_params.devices = devices;
    model_params.n_gpu_layers = offload ? 99 : 0;
    logs.offloaded_layers = 0;
    logs.total_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path, model_params),
        llama_model_free);
    if (!model) {
        throw std::runtime_error(std::string("failed to load model: ") + model_path);
    }
    if (offload && (logs.total_layers == 0 || logs.offloaded_layers != logs.total_layers)) {
        throw std::runtime_error("not all model layers were offloaded to OpenVINO");
    }

    auto params = llama_context_default_params();
    params.n_ctx = 768;
    params.n_seq_max = 3;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    params.kv_unified = false;
    params.type_k = GGML_TYPE_F32;
    params.type_v = GGML_TYPE_F32;
    params.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv = offload;
    params.op_offload = offload;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), params),
                                                                  llama_free);
    if (!context) {
        throw std::runtime_error("failed to create llama context");
    }

    const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    if (vocabulary <= 28) {
        throw std::runtime_error("model vocabulary is too small for the test inputs");
    }
    std::vector<float> result;
    result.reserve(static_cast<size_t>(vocabulary) * 8);
    Batch batch;
    for (int step = 0; step < 4; ++step) {
        if (restore && step == 2) {
            restore_sequences(context.get(), first_seq);
        }
        batch.value.n_tokens = 2;
        for (int seq = 0; seq < 2; ++seq) {
            batch.value.token[seq] = 2 + step * 3 + seq * 17;
            batch.value.pos[seq] = step;
            batch.value.n_seq_id[seq] = 1;
            batch.value.seq_id[seq][0] = first_seq + seq;
            batch.value.logits[seq] = 1;
        }
        if (llama_decode(context.get(), batch.value) != 0) {
            throw std::runtime_error("llama_decode failed");
        }
        for (int seq = 0; seq < 2; ++seq) {
            const float* logits = llama_get_logits_ith(context.get(), seq);
            if (!logits) {
                throw std::runtime_error("missing logits");
            }
            result.insert(result.end(), logits, logits + vocabulary);
        }
    }
    return result;
}

void compare(const char* label, const std::vector<float>& actual, const std::vector<float>& reference) {
    if (actual.empty() || actual.size() != reference.size()) {
        throw std::runtime_error(std::string(label) + ": empty or mismatched logits");
    }
    float max_abs = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) {
            throw std::runtime_error(std::string(label) + ": non-finite logit at index " + std::to_string(i));
        }
        const float delta = std::abs(actual[i] - reference[i]);
        max_abs = std::max(max_abs, delta);
        if (delta > 1e-3f + 1e-3f * std::abs(reference[i])) {
            throw std::runtime_error(std::string(label) + ": logit mismatch at index " + std::to_string(i) +
                                     ", actual=" + std::to_string(actual[i]) +
                                     ", reference=" + std::to_string(reference[i]));
        }
    }
    std::printf("%s: %zu logits, max absolute error %.8g\n", label, actual.size(), max_abs);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 || (std::strcmp(argv[2], "flash") != 0 && std::strcmp(argv[2], "noflash") != 0) ||
        (std::strcmp(argv[3], "0") != 0 && std::strcmp(argv[3], "1") != 0)) {
        std::fprintf(stderr, "usage: %s MODEL {flash|noflash} {0|1}\n", argv[0]);
        return 2;
    }
    try {
        LogState logs;
        llama_log_set(log_message, &logs);
        ggml_backend_load_all();
        ggml_backend_dev_t device = ggml_backend_dev_by_name("OPENVINO0");
        const char* description = device ? ggml_backend_dev_description(device) : nullptr;
        if (!description || !std::strstr(description, "GGML_OPENVINO_DEVICE=CPU (selected)")) {
            throw std::runtime_error("OPENVINO0 is not the selected OpenVINO CPU backend");
        }
        ov_singleton_core().set_property("CPU",
                                         {{"INFERENCE_PRECISION_HINT", ov::element::f32},
                                          {"DYNAMIC_QUANTIZATION_GROUP_SIZE", uint64_t{0}},
                                          {"KV_CACHE_PRECISION", ov::element::f32}});
        const bool flash = std::strcmp(argv[2], "flash") == 0;
        const int first_seq = argv[3][0] - '0';
        const auto cpu = run(argv[1], nullptr, flash, first_seq, false, logs);
        const auto openvino = run(argv[1], device, flash, first_seq, false, logs);
        const auto restored = run(argv[1], device, flash, first_seq, true, logs);
        compare("OpenVINO vs ggml CPU", openvino, cpu);
        compare("OpenVINO after state restore", restored, openvino);
        llama_backend_free();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GGUF llama.cpp integration failed: %s\n", error.what());
        return 1;
    }
}
