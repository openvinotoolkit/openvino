// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "clip-impl.h"
#include "clip.h"
#include "ggml-backend.h"

namespace {
struct Dump {
    std::filesystem::path directory;
    bool waveform = false;
};
template <typename T>
void save(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4)
        return 2;
    const auto directory = std::filesystem::path(argv[3]);
    std::filesystem::create_directories(directory);
    Dump dump{directory};
    ggml_backend_load_all();
    clip_context_params config{};
    config.use_gpu = false;
    config.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    config.cb_eval_user_data = &dump;
    config.cb_eval = [](ggml_tensor* tensor, bool ask, void* data) {
        const std::string name = tensor->name;
        const auto& dump = *static_cast<const Dump*>(data);
        if (ask) {
            const bool selected = dump.waveform
                                      ? name.find("wav_") == 0
                                      : name.find("step_logits-") == 0 || name.find("gen_audio_out") == 0 ||
                                            name.find("step_proj_in") == 0 || name.find("step_layer_out") == 0;
            return selected && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                   name.find("(") == std::string::npos;
        }
        std::vector<char> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        save(dump.directory / (name + ".bin"), bytes);
        return true;
    };
    auto contexts = clip_init(argv[1], config);
    auto* context = contexts.ctx_gen_a;
    if (!context)
        return 3;
    clip_image_f32 image;
    const auto width = clip_n_mmproj_embd(context);
    image.set_size({width, 1}, false, true);
    std::vector<float> hidden(size_t(width), 0.f);
    std::ifstream input(argv[2], std::ios::binary);
    input.read(reinterpret_cast<char*>(hidden.data()), hidden.size() * sizeof(float));
    if (!input)
        return 4;
    image.cpy_buf(hidden);
    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(image);
    clip_encode_params params;
    params.imgs = &batch;
    params.n_threads = 4;
    params.seed = 1729;
    params.code0 = 7;
    params.top_k = 50;
    params.top_p = 1.f;
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    std::vector<float> random(15);
    for (auto& value : random)
        value = distribution(rng);
    save(directory / "random.bin", random);
    std::vector<int32_t> codes;
    std::vector<float> feedback(size_t(width), 0.f);
    params.out_codes = &codes;
    params.out_embd = &feedback;
    params.gen_process = CLIP_GEN_PROCESS_GEN_CODE;
    if (!clip_encode(context, &params))
        return 5;
    save(directory / "codes.bin", codes);
    save(directory / "feedback.bin", feedback);
    std::vector<int32_t> wave_codes(72 * 16);
    for (size_t i = 0; i < wave_codes.size(); ++i)
        wave_codes[i] = int32_t((i * 7 + 3) % 100);
    save(directory / "wave_codes.bin", wave_codes);
    std::vector<uint8_t> state;
    std::vector<float> waveform;
    params.out_codes = nullptr;
    params.out_embd = nullptr;
    params.codes = &wave_codes;
    params.out_audio = &waveform;
    params.state_out = &state;
    params.gen_process = CLIP_GEN_PROCESS_GEN_WAV;
    dump.waveform = true;
    dump.directory = directory / "wave0";
    std::filesystem::create_directories(dump.directory);
    if (!clip_encode(context, &params))
        return 6;
    save(directory / "waveform0.bin", waveform);
    save(directory / "state0.bin", state);
    params.state_in = &state;
    std::vector<uint8_t> second_state;
    params.state_out = &second_state;
    dump.directory = directory / "wave1";
    std::filesystem::create_directories(dump.directory);
    if (!clip_encode(context, &params))
        return 7;
    save(directory / "waveform1.bin", waveform);
    save(directory / "state1.bin", second_state);
    if (contexts.ctx_v)
        clip_free(contexts.ctx_v);
    if (contexts.ctx_a)
        clip_free(contexts.ctx_a);
    clip_free(context);
}
