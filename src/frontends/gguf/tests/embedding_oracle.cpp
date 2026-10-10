// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4)
        return 2;
    llama_backend_init();
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
    auto mp = llama_model_default_params();
    if (!devices[0])
        return 8;
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    auto* m = llama_model_load_from_file(argv[1], mp);
    if (!m)
        return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 64;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* c = llama_init_from_model(m, cp);
    if (!c)
        return 4;
    std::vector<llama_token> tokens{1, 2, 3};
    if (argc == 4) {
        std::string prompt = argv[3];
        tokens.resize(64);
        auto n = llama_tokenize(llama_model_get_vocab(m),
                                prompt.data(),
                                static_cast<int32_t>(prompt.size()),
                                tokens.data(),
                                64,
                                true,
                                false);
        if (n <= 0)
            return 5;
        tokens.resize(n);
    }
    auto batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(c, batch))
        return 6;
    const int32_t dim = llama_model_n_embd(m);
    const int32_t count = static_cast<int32_t>(tokens.size());
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(tokens.data()), sizeof(llama_token) * count);
    for (int i = 0; i < count; ++i)
        out.write(reinterpret_cast<const char*>(llama_get_embeddings_ith(c, i)), dim * sizeof(float));
    llama_free(c);
    llama_model_free(m);
    llama_backend_free();
    return out ? 0 : 7;
}
