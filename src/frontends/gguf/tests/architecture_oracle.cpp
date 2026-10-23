// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Build against a CPU-only llama.cpp. See gen_arch_accuracy.py.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama-ext.h"
#include "llama.h"

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4 && argc != 5)
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
    std::string dump_path;
    if (const char* directory = std::getenv("GGUF_ARCH_ORACLE_DUMP")) {
        std::filesystem::create_directories(directory);
        dump_path = directory;
        cp.cb_eval_user_data = &dump_path;
        cp.cb_eval = [](ggml_tensor* tensor, bool ask, void* data) {
            if (ask)
                return tensor->type == GGML_TYPE_F32;
            std::vector<char> bytes(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
            std::ofstream file(
                std::filesystem::path(*static_cast<const std::string*>(data)) / (std::string(tensor->name) + ".bin"),
                std::ios::binary);
            file.write(bytes.data(), bytes.size());
            return true;
        };
    }
    auto* ctx = llama_init_from_model(model, cp);
    if (!ctx)
        return 4;
    char architecture[64]{};
    llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture));
    const int32_t full_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const bool tts = std::string(architecture) == "qwen3tts";
    const int32_t vocab = tts ? 3072 : full_vocab;
    if (full_vocab < vocab)
        return 12;
    const int32_t vocab_offset = tts ? full_vocab - vocab : 0;
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(&vocab), sizeof(vocab));
    const auto* vocabulary = llama_model_get_vocab(model);
    const bool generation = argc == 4;
    const bool dflash = argc == 5 && std::string(argv[3]) == "--dflash";
    const bool draft = argc == 5 && (std::string(argv[3]) == "--draft" || dflash);
    const bool embeddings = argc == 5 && !draft;
    char lilicorr_metadata[32]{};
    const bool lilicorr =
        llama_model_meta_val_str(model, "lilicorr.hidden_size", lilicorr_metadata, sizeof(lilicorr_metadata)) >= 0;
    char markov_metadata[32]{};
    const bool markov =
        llama_model_meta_val_str(model, "dflash.sample_from_anchor", markov_metadata, sizeof(markov_metadata)) >= 0;
    std::ifstream embedding_file;
    if (embeddings || draft)
        embedding_file.open(argv[4], std::ios::binary);
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
        if (const auto directory = std::getenv("GGUF_ARCH_ORACLE_DUMP")) {
            const auto path = std::filesystem::path(directory) / std::to_string(step);
            std::filesystem::create_directories(path);
            dump_path = path.string();
        }
        schedule << tokens.size();
        for (auto token : tokens)
            schedule << ' ' << token;
        schedule << '\n';
        auto batch = embeddings
                         ? llama_batch_init(static_cast<int32_t>(tokens.size()), llama_model_n_embd_inp(model), 1)
                         : llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        auto* allocated_positions = batch.pos;
        std::vector<llama_pos> positions(4 * tokens.size());
        if (embeddings) {
            batch.n_tokens = static_cast<int32_t>(tokens.size());
            embedding_file.read(reinterpret_cast<char*>(batch.embd),
                                tokens.size() * llama_model_n_embd_inp(model) * sizeof(float));
            if (!embedding_file)
                return 8;
            const int offset = step == 0 ? 0 : step == 1 ? 3 : 4;
            batch.pos = positions.data();
            for (size_t i = 0; i < tokens.size(); ++i) {
                for (size_t section = 0; section < 4; ++section)
                    batch.pos[section * tokens.size() + i] = section < 3 ? offset + static_cast<int>(i) : 0;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = i + 1 == tokens.size();
            }
        }
        if (dflash) {
            if (llama_model_dflash_selector_top_k(model) > 0 || lilicorr || markov)
                llama_set_embeddings_nextn(ctx, true, false);
            const size_t width = 3 * llama_model_n_embd(model);
            std::vector<float> features(tokens.size() * width);
            embedding_file.read(reinterpret_cast<char*>(features.data()), features.size() * sizeof(float));
            if (!embedding_file)
                return 8;
            const int offset = lilicorr ? step * 3 : step == 0 ? 0 : step == 1 ? 3 : 4;
            auto* injection = llama_batch_ext_init(ctx);
            for (size_t i = 0; i < tokens.size(); ++i) {
                const auto index = llama_batch_ext_add_embd(injection, 0, {features.data() + i * width, 1, width});
                const llama_pos position = offset + i;
                llama_batch_ext_set_pos(injection, index, &position);
                llama_batch_ext_set_output_embd(injection, index, true);
            }
            llama_set_causal_attn(ctx, false);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, injection))
                return 9;
            std::vector<float> anchors;
            if (lilicorr) {
                anchors.assign(llama_get_embeddings_nextn(ctx),
                               llama_get_embeddings_nextn(ctx) + tokens.size() * llama_model_n_embd(model));
                std::ofstream anchor_file(std::string(argv[2]) + ".anchor" + std::to_string(step), std::ios::binary);
                anchor_file.write(reinterpret_cast<const char*>(anchors.data()), anchors.size() * sizeof(float));
            }
            llama_batch_ext_free(injection);
            auto* decoder = llama_batch_ext_init(ctx);
            for (size_t i = 0; i < tokens.size(); ++i) {
                const auto index = llama_batch_ext_add_token(decoder, 0, tokens[i]);
                if (lilicorr)
                    llama_batch_ext_set_embd_state(decoder,
                                                   index,
                                                   {anchors.data() + (tokens.size() - 1) * llama_model_n_embd(model),
                                                    1,
                                                    size_t(llama_model_n_embd(model))});
                const llama_pos position = offset + tokens.size() + i;
                llama_batch_ext_set_pos(decoder, index, &position);
                llama_batch_ext_set_output_logits(decoder, index, true);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, decoder))
                return 10;
            if (markov) {
                std::ofstream all_logits(std::string(argv[2]) + ".alllogits" + std::to_string(step), std::ios::binary);
                all_logits.write(reinterpret_cast<const char*>(llama_get_logits(ctx)),
                                 tokens.size() * llama_vocab_n_tokens(llama_model_get_vocab(model)) * sizeof(float));
            }
            if (llama_model_dflash_selector_top_k(model) > 0 || lilicorr || markov) {
                std::ofstream lattice(std::string(argv[2]) + ".lattice" + std::to_string(step), std::ios::binary);
                lattice.write(reinterpret_cast<const char*>(llama_get_embeddings_nextn(ctx)),
                              tokens.size() * llama_model_n_embd(model) * sizeof(float));
            }
            llama_batch_ext_free(decoder);
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, offset + tokens.size(), -1))
                return 11;
        } else if (draft) {
            const size_t width = 3 * llama_model_n_embd(model);
            const size_t hidden = llama_model_n_embd(model);
            std::vector<float> features(tokens.size() * width);
            embedding_file.read(reinterpret_cast<char*>(features.data()), features.size() * sizeof(float));
            if (!embedding_file)
                return 8;
            llama_set_embeddings_nextn(ctx, true, false);
            auto* encoder = llama_batch_ext_init(ctx);
            const int offset = step == 0 ? 0 : step == 1 ? 3 : 4;
            for (size_t i = 0; i < tokens.size(); ++i) {
                const auto index = llama_batch_ext_add_embd(encoder, 0, {features.data() + i * width, 1, width});
                const llama_pos position = offset + i;
                llama_batch_ext_set_pos(encoder, index, &position);
                llama_batch_ext_set_output_embd(encoder, index, true);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_ENCODE, encoder))
                return 9;
            std::vector<float> fused(llama_get_embeddings_nextn(ctx),
                                     llama_get_embeddings_nextn(ctx) + tokens.size() * hidden);
            llama_batch_ext_free(encoder);
            auto* decoder = llama_batch_ext_init(ctx);
            for (size_t i = 0; i < tokens.size(); ++i) {
                const auto index = llama_batch_ext_add_token(decoder, 0, tokens[i]);
                llama_batch_ext_set_embd_token(decoder, index, {fused.data() + i * hidden, 1, hidden});
                const llama_pos position = offset + i;
                llama_batch_ext_set_pos(decoder, index, &position);
                llama_batch_ext_set_output_logits(decoder, index, true);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, decoder))
                return 10;
            llama_batch_ext_free(decoder);
        } else if (llama_decode(ctx, batch))
            return 5;
        if (embeddings) {
            batch.pos = allocated_positions;
            llama_batch_free(batch);
        }
        const auto* logits = llama_get_logits_ith(ctx, -1) + vocab_offset;
        if (tts && !std::all_of(logits, logits + vocab, [](float value) {
                return std::isfinite(value);
            }))
            return 13;
        out.write(reinterpret_cast<const char*>(logits), vocab * sizeof(float));
        tokens = lilicorr     ? std::vector<llama_token>{4, 5, 6}
                 : generation ? std::vector<llama_token>{static_cast<llama_token>(
                                    std::max_element(logits, logits + vocab) - logits + vocab_offset)}
                 : step == 0  ? std::vector<llama_token>{4}
                              : std::vector<llama_token>{5, 6};
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return out ? 0 : 6;
}
