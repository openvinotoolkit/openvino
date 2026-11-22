// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>

#include "ggml-backend.h"
#include "llama-ext.h"
#include "llama.h"

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    llama_backend_init();
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
    if (!devices[0])
        return 3;
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    auto* model = llama_model_load_from_file(argv[1], mp);
    if (!model)
        return 4;
    llama_dspark_meta meta;
    if (!llama_model_dspark_get_meta(model, &meta))
        return 5;
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = 128;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* ctx = llama_init_from_model(model, cp);
    if (!ctx)
        return 6;
    llama_set_causal_attn(ctx, false);
    std::ofstream out(argv[2], std::ios::binary);
    const int32_t dimensions[] = {static_cast<int32_t>(meta.n_embd_cap),
                                  meta.block_size,
                                  static_cast<int32_t>(meta.n_vocab)};
    out.write(reinterpret_cast<const char*>(dimensions), sizeof(dimensions));
    for (int32_t rows : {2, 3}) {
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<float> features(rows * meta.n_embd_cap);
        std::vector<int32_t> positions(rows);
        for (size_t i = 0; i < features.size(); ++i)
            features[i] = std::sin(static_cast<float>(i)) * .1f;
        for (int32_t i = 0; i < rows; ++i)
            positions[i] = i;
        llama_set_dspark_ctx(ctx, features.data(), rows, meta.n_embd_cap, positions.data());
        auto batch = llama_batch_init(rows + meta.block_size, 0, 1);
        batch.n_tokens = rows + meta.block_size;
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = i < rows ? 0 : i - rows + 1;
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i >= rows;
        }
        const auto rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc)
            return 7;
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(llama_get_logits(ctx)), meta.block_size * meta.n_vocab * sizeof(float));
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return out ? 0 : 8;
}
