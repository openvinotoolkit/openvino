// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "clip-impl.h"
#include "clip.h"
#include "ggml-backend.h"

// model modality output-directory [width height input.f32 second.f32|-]...
int main(int argc, char** argv) {
    if (argc < 8 || (argc - 4) % 4)
        return 2;
    ggml_backend_load_all();
    std::filesystem::path dump;
    clip_context_params params{};
    params.use_gpu = false;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    params.cb_eval_user_data = &dump;
    params.cb_eval = [](ggml_tensor* tensor, bool ask, void* data) {
        if (ask)
            return tensor->type == GGML_TYPE_F32 && std::string(tensor->name).rfind("state_out_", 0) == 0;
        std::vector<char> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        auto path = *static_cast<std::filesystem::path*>(data) / (std::string(tensor->name) + ".f32");
        std::ofstream file(path, std::ios::binary);
        file.write(bytes.data(), bytes.size());
        return bool(file);
    };
    const auto contexts = clip_init(argv[1], params);
    const bool audio = std::string(argv[2]) == "audio";
    auto* context = audio ? contexts.ctx_a : contexts.ctx_v;
    if (!context)
        return 3;
    auto* state = audio ? clip_enc_state_init(context, 128) : nullptr;
    if (audio && !state)
        return 4;
    for (int i = 4; i < argc; i += 4) {
        dump = std::filesystem::path(argv[3]) / std::to_string((i - 4) / 4);
        std::filesystem::create_directories(dump);
        const int width = std::stoi(argv[i]), height = std::stoi(argv[i + 1]);
        std::vector<float> values(size_t(width) * height * (audio ? 1 : 3));
        auto read = [&](const char* name) {
            std::ifstream file(name, std::ios::binary);
            file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
            return bool(file);
        };
        if (!read(argv[i + 2]))
            return 5;
        clip_image_f32 image;
        image.set_size({width, height}, false, audio);
        image.cpy_buf(values);
        clip_image_f32_batch batch;
        batch.is_audio = audio;
        batch.entries.push_back(image);
        if (std::string(argv[i + 3]) != "-") {
            if (!read(argv[i + 3]))
                return 5;
            image.cpy_buf(values);
            batch.entries.push_back(image);
        }
        std::vector<float> output(size_t(clip_n_output_tokens(context, &image)) * clip_n_mmproj_embd(context));
        clip_encode_params encode;
        encode.n_threads = 2;
        encode.imgs = &batch;
        encode.out_embd = &output;
        encode.enc_state = state;
        if (!clip_encode(context, &encode))
            return 6;
        std::ofstream file(dump / "embeddings.f32", std::ios::binary);
        file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
    }
    if (state)
        clip_enc_state_free(state);
    if (contexts.ctx_v)
        clip_free(contexts.ctx_v);
    if (contexts.ctx_a)
        clip_free(contexts.ctx_a);
    if (contexts.ctx_gen_a)
        clip_free(contexts.ctx_gen_a);
}
