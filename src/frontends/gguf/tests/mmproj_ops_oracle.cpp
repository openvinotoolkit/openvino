// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Offline geometry oracle. Link to the pinned ggml CPU libraries, then pass an output directory.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    auto* ctx = ggml_init({64 * 1024 * 1024, nullptr, false});
    auto* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 5, 3, 1);
    for (int i = 0; i < 45; ++i)
        static_cast<float*>(input->data)[i] = std::sin(float(i) * .13f);
    auto* windows = ggml_win_part(ctx, input, 2);
    auto* restored = ggml_win_unpart(ctx, windows, 5, 3, 2);
    auto* table = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4, 5);
    for (int i = 0; i < 20; ++i)
        static_cast<ggml_fp16_t*>(table->data)[i] = ggml_fp32_to_fp16(std::sin(float(i) * .13f));
    auto* relative = ggml_get_rel_pos(ctx, table, 3, 3);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, restored);
    ggml_build_forward_expand(graph, relative);
    if (ggml_graph_compute_with_ctx(ctx, graph, 2) != GGML_STATUS_SUCCESS)
        return 3;
    std::filesystem::create_directories(argv[1]);
    const auto save = [&](const char* name, ggml_tensor* t) {
        std::vector<float> data(size_t(ggml_nelements(t)));
        if (t->type == GGML_TYPE_F32)
            std::copy_n(static_cast<float*>(t->data), data.size(), data.data());
        else
            for (size_t i = 0; i < data.size(); ++i)
                data[i] = ggml_fp16_to_fp32(static_cast<ggml_fp16_t*>(t->data)[i]);
        std::ofstream out(std::filesystem::path(argv[1]) / name, std::ios::binary);
        out.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    };
    save("window_input.bin", input);
    save("windows.bin", windows);
    save("restored.bin", restored);
    save("relative_input.bin", table);
    save("relative.bin", relative);
    ggml_free(ctx);
}
