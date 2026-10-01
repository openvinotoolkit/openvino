// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Offline geometry oracle. Link to the pinned ggml CPU libraries, then pass an output directory.
#include <cmath>
#include <filesystem>
#include <fstream>

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
    // SAM decomposed relative positions: resize a 3-entry [L, C] table to 2 * 3 - 1 rows, gather q - k + 2.
    auto* table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    for (int i = 0; i < 12; ++i)
        static_cast<float*>(table->data)[i] = std::sin(float(i) * .37f);
    auto* indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 9);
    for (int q = 0; q < 3; ++q)
        for (int k = 0; k < 3; ++k)
            static_cast<int32_t*>(indices->data)[q * 3 + k] = q - k + 2;
    auto* lengths = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, table)), 3, 1, 4);
    auto* resized = ggml_cont(
        ctx,
        ggml_transpose(
            ctx,
            ggml_reshape_2d(ctx, ggml_interpolate(ctx, lengths, 5, 1, 4, 1, GGML_SCALE_MODE_BILINEAR), 5, 4)));
    auto* relative = ggml_get_rows(ctx, resized, indices);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, restored);
    ggml_build_forward_expand(graph, relative);
    if (ggml_graph_compute_with_ctx(ctx, graph, 2) != GGML_STATUS_SUCCESS)
        return 3;
    std::filesystem::create_directories(argv[1]);
    const auto save = [&](const char* name, ggml_tensor* t) {
        std::ofstream out(std::filesystem::path(argv[1]) / name, std::ios::binary);
        out.write(static_cast<char*>(t->data), ggml_nbytes(t));
    };
    save("window_input.bin", input);
    save("windows.bin", windows);
    save("restored.bin", restored);
    save("relative_table.bin", table);
    save("relative.bin", relative);
    ggml_free(ctx);
}
