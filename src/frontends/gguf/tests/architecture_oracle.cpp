// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Build against a CPU-only llama.cpp. See gen_arch_accuracy.py.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4)
        return 2;
    llama_backend_init();
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
    if (!devices[0])
        throw std::runtime_error("The architecture oracle requires the ggml CPU backend");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    auto* model = llama_model_load_from_file(argv[1], mp);
    if (!model)
        return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* ctx = llama_init_from_model(model, cp);
    if (!ctx)
        return 4;
    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(&vocab), sizeof(vocab));
    const auto* vocabulary = llama_model_get_vocab(model);
    const bool generation = argc == 4;
    std::vector<llama_token> tokens{1, 2, 3};
    if (generation) {
        const std::string prompt = argv[3];
        tokens.resize(64);
        const auto count = llama_tokenize(vocabulary,
                                          prompt.data(),
                                          static_cast<int32_t>(prompt.size()),
                                          tokens.data(),
                                          64,
                                          true,
                                          false);
        if (count <= 0)
            return 7;
        tokens.resize(count);
    }
    std::ofstream schedule(std::string(argv[2]) + ".tokens");
    for (int step = 0; step < (generation ? 13 : 3); ++step) {
        schedule << tokens.size();
        for (auto token : tokens)
            schedule << ' ' << token;
        schedule << '\n';
        auto batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        if (llama_decode(ctx, batch))
            return 5;
        const auto* logits = llama_get_logits_ith(ctx, -1);
        out.write(reinterpret_cast<const char*>(logits), vocab * sizeof(float));
        tokens =
            generation
                ? std::vector<llama_token>{static_cast<llama_token>(std::max_element(logits, logits + vocab) - logits)}
            : step == 0 ? std::vector<llama_token>{4}
                        : std::vector<llama_token>{5, 6};
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return out ? 0 : 6;
}
