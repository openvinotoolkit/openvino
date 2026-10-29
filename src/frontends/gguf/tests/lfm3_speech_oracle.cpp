// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include "clip-impl.h"
#include "clip.h"
#include "ggml-backend.h"

namespace {
template <class T>
std::vector<T> read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("cannot read " + path.string());
    const std::streamoff size = file.tellg();
    if (size < 0)
        throw std::runtime_error("invalid input size " + path.string());
    const auto bytes = static_cast<uintmax_t>(size);
    if (bytes % sizeof(T) != 0 || bytes > std::numeric_limits<size_t>::max() ||
        bytes > static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("invalid input size " + path.string());
    std::vector<T> values(static_cast<size_t>(bytes) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (!file)
        throw std::runtime_error("short input " + path.string());
    return values;
}
void write(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    if (!file)
        throw std::runtime_error("cannot write " + path.string());
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4)
        return 2;
    ggml_backend_load_all();
    clip_context_params options{};
    options.use_gpu = false;
    options.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    const auto contexts = clip_init(argv[1], options);
    if (!contexts.ctx_gen_a)
        return 3;
    const auto input = std::filesystem::path(argv[2]);
    const auto output = std::filesystem::path(argv[3]);
    std::filesystem::create_directories(output);
    const auto hidden = read<float>(input / "hidden.f32");
    const auto codes = read<int32_t>(input / "codes.i32");
    const auto depth_codes = read<int32_t>(input / "depth_codes.i32");
    const int width = clip_n_mmproj_embd(contexts.ctx_gen_a);
    clip_image_f32 image;
    image.set_size({1, width}, false, false);
    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(image);
    auto* state = clip_gen_state_init(contexts.ctx_gen_a, 42);
    if (!state)
        return 4;
    clip_encode_params params;
    params.n_threads = 2;
    params.imgs = &batch;
    params.gen_state = state;
    std::vector<float> feedback(width), features;
    params.out_embd = &feedback;
    params.out_feats = &features;
    params.gen_process = CLIP_GEN_PROCESS_GEN_CODE;
    for (int phase = 0; phase < 3; ++phase) {
        for (size_t frame = 0; frame < hidden.size() / width; ++frame) {
            batch.entries[0].cpy_buf(
                std::vector<float>(hidden.begin() + frame * width, hidden.begin() + (frame + 1) * width));
            std::vector<int32_t> current(depth_codes.begin() + frame * 12, depth_codes.begin() + (frame + 1) * 12);
            for (size_t book = 0; book < current.size(); ++book)
                if (phase == 0 || (phase == 1 && book % 3 != 0))
                    current[book] = -1;
            params.codes = &current;
            features.clear();
            if (!clip_encode(contexts.ctx_gen_a, &params))
                return 5;
            const auto stem = "depth_" + std::to_string(phase) + "_" + std::to_string(frame);
            if (phase < 2)
                write(output / (stem + ".f32"), features);
            else
                write(output / ("feedback_" + std::to_string(frame) + ".f32"), feedback);
        }
    }
    params.gen_process = CLIP_GEN_PROCESS_GEN_WAV;
    params.out_embd = nullptr;
    const int counts[] = {1, 2, 6, 2, 9};
    size_t offset = 0;
    for (int step = 0; step < 5; ++step) {
        if (step >= 3) {
            clip_gen_state_reset(state);
            offset = step == 3 ? 1 : 0;
        }
        std::vector<int32_t> current(codes.begin() + offset * 12, codes.begin() + (offset + counts[step]) * 12);
        params.codes = &current;
        if (!clip_encode(contexts.ctx_gen_a, &params))
            return 6;
        write(output / ("spectra_" + std::to_string(step) + ".f32"), features);
        for (const auto* slot : state->slots) {
            std::vector<float> values(ggml_nelements(slot));
            ggml_backend_tensor_get(slot, values.data(), 0, ggml_nbytes(slot));
            write(output / ("state_" + std::to_string(step) + "_" + slot->name + ".f32"), values);
        }
        offset += counts[step];
    }
    clip_gen_state_free(state);
    clip_free(contexts.ctx_gen_a);
    if (contexts.ctx_a)
        clip_free(contexts.ctx_a);
    if (contexts.ctx_v)
        clip_free(contexts.ctx_v);
    return 0;
}
