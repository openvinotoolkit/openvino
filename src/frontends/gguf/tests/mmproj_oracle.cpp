// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Offline reference generator. Link against the pinned llama.cpp libmtmd and libggml.
// Usage: mmproj_oracle model.gguf vision|audio width height input.f32 output.f32
#include <fstream>
#include <iostream>

#include "clip-impl.h"
#include "clip.h"
#include "ggml-backend.h"

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8)
        return 2;
    ggml_backend_load_all();
    clip_context_params params{};
    params.use_gpu = false;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    const auto contexts = clip_init(argv[1], params);
    const bool audio = std::string(argv[2]) == "audio";
    auto* context = audio ? contexts.ctx_a : contexts.ctx_v;
    if (!context)
        return 3;
    clip_image_f32 input;
    const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
    input.set_size({width, height}, false, audio);
    std::vector<float> values(size_t(width) * height * (audio ? 1 : 3));
    std::ifstream file(argv[5], std::ios::binary);
    file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
    if (!file)
        return 4;
    input.cpy_buf(values);
    std::vector<float> output(size_t(clip_n_output_tokens(context, &input)) * clip_n_mmproj_embd(context));
    clip_image_f32_batch batch;
    batch.is_audio = audio;
    batch.entries.push_back(input);
    if (argc == 8) {
        std::ifstream second(argv[7], std::ios::binary);
        second.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
        if (!second)
            return 4;
        input.cpy_buf(values);
        batch.entries.push_back(input);
    }
    const bool ok = clip_image_batch_encode(context, 2, &batch, output);
    if (ok) {
        std::ofstream result(argv[6], std::ios::binary);
        result.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
    }
    if (contexts.ctx_v)
        clip_free(contexts.ctx_v);
    if (contexts.ctx_a)
        clip_free(contexts.ctx_a);
    return ok ? 0 : 5;
}
