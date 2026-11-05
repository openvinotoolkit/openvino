// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// CPU-only reference; see gen_mtp_accuracy.py for the pinned llama.cpp revision.
#include <cmath>
#include <fstream>
#include <stdexcept>
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
        throw std::runtime_error("MTP oracle requires the ggml CPU backend");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    auto* model = llama_model_load_from_file(argv[1], mp);
    if (!model)
        return 3;
    auto cp = llama_context_default_params();
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 64;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* ctx = llama_init_from_model(model, cp);
    if (!ctx)
        return 4;
    llama_set_embeddings_nextn(ctx, true, true);
    auto* batch = llama_batch_ext_init(ctx);
    const auto width = llama_model_n_embd(model);
    const auto vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ofstream out(argv[2], std::ios::binary);
    int32_t offset = 0;
    for (const auto& tokens : {std::vector<llama_token>{1, 2, 3}, {4}, {5, 6}}) {
        llama_batch_ext_clear(batch);
        std::vector<float> hidden(tokens.size() * width);
        for (size_t i = 0; i < hidden.size(); ++i)
            hidden[i] = 0.2f * std::sin(static_cast<float>(offset * width + i + 1));
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
            const llama_pos pos = offset + i;
            if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, &pos) ||
                !llama_batch_ext_set_embd_state(batch, idx, {hidden.data() + i * width, 1, size_t(width)}) ||
                !llama_batch_ext_set_output_logits(batch, idx, true))
                return 5;
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch))
            return 6;
        out.write(reinterpret_cast<const char*>(hidden.data()), hidden.size() * sizeof(float));
        for (size_t i = 0; i < tokens.size(); ++i) {
            out.write(reinterpret_cast<const char*>(llama_get_logits_ith(ctx, i)), vocab * sizeof(float));
            out.write(reinterpret_cast<const char*>(llama_get_embeddings_nextn_ith(ctx, i)), width * sizeof(float));
        }
        offset += tokens.size();
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return out ? 0 : 7;
}
