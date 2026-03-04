// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fused_mlp.hpp>
#include <intel_gpu/primitives/input_layout.hpp>

#include <cmath>
#include <numeric>
#include <vector>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

#ifdef ENABLE_ONEDNN_FOR_GPU

namespace {

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static std::vector<float> to_f32(const std::vector<ov::float16>& v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        out[i] = static_cast<float>(v[i]);
    return out;
}

static std::vector<ov::float16> make_f16_values(size_t count, float scale) {
    std::vector<ov::float16> v(count);
    for (size_t i = 0; i < count; ++i) {
        float x = (static_cast<int>(i % 23) - 11) * scale;
        v[i] = ov::float16(x);
    }
    return v;
}

static std::vector<float> matmul(const std::vector<float>& a, const std::vector<float>& b, int64_t m, int64_t k, int64_t n) {
    std::vector<float> c(static_cast<size_t>(m * n), 0.0f);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int64_t kk = 0; kk < k; ++kk) {
                acc += a[static_cast<size_t>(i * k + kk)] * b[static_cast<size_t>(kk * n + j)];
            }
            c[static_cast<size_t>(i * n + j)] = acc;
        }
    }
    return c;
}

}  // namespace

TEST(fused_mlp_onednn_graph_gpu, smoke_fp16) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        return;
    }

    // clDNN tensor storage is (b, f, x, y), while format::bfyx sizes() order is (b, f, y, x).
    // Encode X [B, S, IC] as tensor(b=B, f=S, x=1, y=IC) => sizes(bfyx) = [B, S, IC, 1].
    // Weights are encoded as 2D matrices: tensor(b=IC/OC, f=OC/IC, x=1, y=1).
    const int64_t b = 1;
    const int64_t s = 2;
    const int64_t ic = 8;
    const int64_t oc = 16;

    const int64_t mb = b * s;

    auto x_vals = make_f16_values(static_cast<size_t>(b * s * ic), 0.05f);
    auto w_gate_vals = make_f16_values(static_cast<size_t>(ic * oc), 0.02f);
    auto w_up_vals = make_f16_values(static_cast<size_t>(ic * oc), 0.03f);
    auto w_down_vals = make_f16_values(static_cast<size_t>(oc * ic), 0.04f);

    auto x_mem = engine.allocate_memory({data_types::f16, format::bfyx, cldnn::tensor(cldnn::batch(b), cldnn::feature(s), cldnn::spatial(1, ic))});
    auto w_gate_mem = engine.allocate_memory({data_types::f16, format::bfyx, {ic, oc, 1, 1}});
    auto w_up_mem = engine.allocate_memory({data_types::f16, format::bfyx, {ic, oc, 1, 1}});
    auto w_down_mem = engine.allocate_memory({data_types::f16, format::bfyx, {oc, ic, 1, 1}});

    set_values(x_mem, x_vals);
    set_values(w_gate_mem, w_gate_vals);
    set_values(w_up_mem, w_up_vals);
    set_values(w_down_mem, w_down_vals);
    get_test_stream().finish();

    topology t;
    t.add(input_layout("x", x_mem->get_layout()));
    t.add(data("w_gate", w_gate_mem));
    t.add(data("w_up", w_up_mem));
    t.add(data("w_down", w_down_mem));
    t.add(fused_mlp("fused_mlp", {input_info("x"), input_info("w_gate"), input_info("w_up"), input_info("w_down")}));

    ExecutionConfig config = get_test_default_config(engine);
    // Unit tests build clDNN topology directly, so model-driven config finalization does not enable optimize_data.
    // oneDNN implementations are filtered out unless optimize_data is enabled.
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::use_onednn(true));

    network net(engine, t, config);
    net.set_input_data("x", x_mem);

    auto outputs = net.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fused_mlp");

    auto out_mem = outputs.begin()->second.get_memory();
    get_test_stream().flush();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> out_ptr(out_mem, get_test_stream());

    auto out_layout = out_mem->get_layout();
    EXPECT_EQ(out_layout.batch(), b);
    EXPECT_EQ(out_layout.feature(), s);

    // Reference:
    // gate = X * W_gate
    // up   = X * W_up
    // swish = gate * sigmoid(gate)
    // hidden = swish * up
    // Y = hidden * W_down
    auto x_f = to_f32(x_vals);
    auto w_gate_f = to_f32(w_gate_vals);
    auto w_up_f = to_f32(w_up_vals);
    auto w_down_f = to_f32(w_down_vals);

    auto gate = matmul(x_f, w_gate_f, mb, ic, oc);
    auto up = matmul(x_f, w_up_f, mb, ic, oc);

    std::vector<float> swish(static_cast<size_t>(mb * oc), 0.0f);
    for (size_t i = 0; i < swish.size(); ++i) {
        swish[i] = gate[i] * sigmoid(gate[i]);
    }

    std::vector<float> hidden(static_cast<size_t>(mb * oc), 0.0f);
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = swish[i] * up[i];
    }

    auto y = matmul(hidden, w_down_f, mb, oc, ic);

    for (size_t i = 0; i < y.size(); ++i) {
        printf("FusedMLP ULT result: i = %d, gpu = %f, ref = %f\n", i, static_cast<float>(out_ptr[i]), y[i]);
        EXPECT_NEAR(static_cast<float>(out_ptr[i]), y[i], 5e-2f);
    }
}

#else

TEST(fused_mlp_onednn_graph_gpu, smoke_fp16) {
    GTEST_SKIP() << "ENABLE_ONEDNN_FOR_GPU is not enabled";
}

#endif
