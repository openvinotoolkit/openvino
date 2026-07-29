// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/gated_mlp.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "network_test.h"
#include "test_utils.h"
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#ifdef ENABLE_ONEDNN_FOR_GPU

using namespace cldnn;
using namespace ::tests;

namespace {

float swish(float x) {
    return x / (1.0f + std::exp(-x));
}

std::vector<ov::float16> to_f16(const std::vector<float>& src) {
    std::vector<ov::float16> dst;
    dst.reserve(src.size());
    for (auto v : src) {
        dst.push_back(ov::float16(v));
    }
    return dst;
}

std::vector<float> gated_mlp_reference(const std::vector<float>& src,
                                       const std::vector<float>& w_gate,
                                       const std::vector<float>& w_up,
                                       const std::vector<float>& w_down,
                                       int batch,
                                       int ifm,
                                       int hidden) {
    std::vector<float> gate(batch * hidden, 0.0f);
    std::vector<float> up(batch * hidden, 0.0f);
    std::vector<float> out(batch * ifm, 0.0f);

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < hidden; h++) {
            float gate_acc = 0.0f;
            float up_acc = 0.0f;
            for (int i = 0; i < ifm; i++) {
                gate_acc += src[b * ifm + i] * w_gate[i * hidden + h];
                up_acc += src[b * ifm + i] * w_up[i * hidden + h];
            }
            gate[b * hidden + h] = swish(gate_acc);
            up[b * hidden + h] = up_acc;
        }
    }

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < ifm; i++) {
            float acc = 0.0f;
            for (int h = 0; h < hidden; h++) {
                acc += (gate[b * hidden + h] * up[b * hidden + h]) * w_down[h * ifm + i];
            }
            out[b * ifm + i] = acc;
        }
    }

    return out;
}

template <typename QType, typename ZpType>
std::vector<float> dequantize_weights(const std::vector<QType>& qweights,
                                      int rows,
                                      int cols,
                                      const std::vector<float>& scales,
                                      int scale_rows,
                                      int scale_cols,
                                      const std::vector<ZpType>& zps,
                                      int zp_rows,
                                      int zp_cols) {
    std::vector<float> weights(rows * cols, 0.0f);
    const int scale_group_size = rows / scale_rows;
    const int zp_group_size = rows / zp_rows;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const int scale_row = r / scale_group_size;
            const int scale_col = (scale_cols == 1 ? 0 : c);
            const int zp_row = r / zp_group_size;
            const int zp_col = (zp_cols == 1 ? 0 : c);

            const float scale = scales[scale_row * scale_cols + scale_col];
            const int zp = static_cast<int>(zps[zp_row * zp_cols + zp_col]);
            const int q = static_cast<int>(qweights[r * cols + c]);
            weights[r * cols + c] = static_cast<float>(q - zp) * scale;
        }
    }

    return weights;
}

void check_output(memory::ptr output_mem, const std::vector<float>& ref, float tol = 1e-2f) {
    cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());
    ASSERT_EQ(output_ptr.size(), ref.size());

    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_NEAR(static_cast<float>(output_ptr[i]), ref[i], tol) << "idx=" << i;
    }
}

void check_output_finite(memory::ptr output_mem) {
    cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_TRUE(std::isfinite(static_cast<float>(output_ptr[i]))) << "idx=" << i;
    }
}



std::shared_ptr<network> create_fp16_network(engine& engine,
                                             int batch,
                                             int ifm,
                                             int hidden,
                                             memory::ptr src,
                                             memory::ptr w_gate,
                                             memory::ptr w_up,
                                             memory::ptr w_down) {
    topology topology(
        input_layout("src", layout({batch, 1, 1, ifm}, data_types::f16, format::bfyx)),
        data("w_gate", w_gate),
        data("w_up", w_up),
        data("w_down", w_down),
        reorder("src_2d", input_info("src"), {data_types::f16, format::bfyx, tensor(batch, ifm, 1, 1)}),
        gated_mlp("gmlp",
                  input_info("src_2d"),
                  input_info("w_gate"),
                  input_info("w_up"),
                  input_info("w_down"),
                  ov::op::internal::GLU::GluType::Swish,
                  tensor(batch, ifm, 1, 1),
                  data_types::f16)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::use_onednn(true));

    auto net = std::make_shared<network>(engine, topology, config);
    net->set_input_data("src", src);
    return net;
}

std::shared_ptr<network> create_compressed_network(engine& engine,
                                                   int batch,
                                                   int ifm,
                                                   int hidden,
                                                   memory::ptr src,
                                                   memory::ptr w_gate,
                                                   memory::ptr w_up,
                                                   memory::ptr w_down,
                                                   memory::ptr s_gate,
                                                   memory::ptr s_up,
                                                   memory::ptr s_down,
                                                   memory::ptr zp_gate,
                                                   memory::ptr zp_up,
                                                    memory::ptr zp_down,
                                                    const std::string& prim_name = "gmlp") {
    topology topology(
        input_layout("src", layout({batch, 1, 1, ifm}, data_types::f16, format::bfyx)),
        data("w_gate", w_gate),
        data("w_up", w_up),
        data("w_down", w_down),
        data("s_gate", s_gate),
        data("s_up", s_up),
        data("s_down", s_down),
        data("zp_gate", zp_gate),
        data("zp_up", zp_up),
        data("zp_down", zp_down),
        reorder("src_2d", input_info("src"), {data_types::f16, format::bfyx, tensor(batch, ifm, 1, 1)}),
        gated_mlp(prim_name,
                  input_info("src_2d"),
                  input_info("w_gate"),
                  input_info("w_up"),
                  input_info("w_down"),
                  input_info("s_gate"),
                  input_info("s_up"),
                  input_info("s_down"),
                  input_info("zp_gate"),
                  input_info("zp_up"),
                  input_info("zp_down"),
                  ov::op::internal::GLU::GluType::Swish,
                  tensor(batch, ifm, 1, 1),
                  data_types::f16)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::use_onednn(true));

    auto net = std::make_shared<network>(engine, topology, config);
    net->set_input_data("src", src);
    return net;
}

}  // namespace

// ============================================================
// FP16 Parameterized Tests
// ============================================================
struct FP16Params {
    int batch;
    int ifm;
    int hidden;
};

class GatedMlpGPU_FP16 : public ::testing::TestWithParam<FP16Params> {};

TEST_P(GatedMlpGPU_FP16, basic) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "gated_mlp oneDNN implementation requires IMMAD support";
    }
    const auto& p = GetParam();
    const int batch = p.batch, ifm = p.ifm, hidden = p.hidden;

    auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
    auto gate_mem = engine.allocate_memory({{ifm, hidden}, data_types::f16, format::bfyx});
    auto up_mem = engine.allocate_memory({{ifm, hidden}, data_types::f16, format::bfyx});
    auto down_mem = engine.allocate_memory({{hidden, ifm}, data_types::f16, format::bfyx});

    std::vector<float> src(batch * ifm);
    std::vector<float> gate_w(ifm * hidden);
    std::vector<float> up_w(ifm * hidden);
    std::vector<float> down_w(hidden * ifm);

    for (size_t i = 0; i < src.size(); i++)
        src[i] = 0.1f * static_cast<float>(static_cast<int>(i % 7) - 3);
    for (size_t i = 0; i < gate_w.size(); i++)
        gate_w[i] = 0.1f * static_cast<float>(static_cast<int>(i % 11) - 5);
    for (size_t i = 0; i < up_w.size(); i++)
        up_w[i] = 0.1f * static_cast<float>(static_cast<int>(i % 9) - 4);
    for (size_t i = 0; i < down_w.size(); i++)
        down_w[i] = 0.1f * static_cast<float>(static_cast<int>(i % 13) - 6);

    set_values(src_mem, to_f16(src));
    set_values(gate_mem, to_f16(gate_w));
    set_values(up_mem, to_f16(up_w));
    set_values(down_mem, to_f16(down_w));

    auto ref = gated_mlp_reference(src, gate_w, up_w, down_w, batch, ifm, hidden);
    auto net = create_fp16_network(engine, batch, ifm, hidden, src_mem, gate_mem, up_mem, down_mem);
    auto outputs = net->execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "gmlp");
    check_output(outputs.at("gmlp").get_memory(), ref);
}

INSTANTIATE_TEST_SUITE_P(
    gated_mlp_gpu,
    GatedMlpGPU_FP16,
    ::testing::Values(
        FP16Params{2, 4, 3},
        FP16Params{1, 3, 4}
    ),
    [](const ::testing::TestParamInfo<FP16Params>& info) {
        const auto& p = info.param;
        return "batch" + std::to_string(p.batch) + "_ifm" + std::to_string(p.ifm) + "_hidden" + std::to_string(p.hidden);
    }
);

// ============================================================
// Compressed Parameterized Tests
// ============================================================
enum class ScaleConfig { PerTensor, Grouped, PerOC };
enum class VerifyMode { Reference, Zero, Finite };

struct CompressedParams {
    int batch;
    int ifm;
    int hidden;
    data_types weight_dt;
    data_types zp_dt;
    ScaleConfig scale_config;
    int scale_groups;
    bool zero_scale;
    float tolerance;
    VerifyMode verify;
};

class GatedMlpGPU_Compressed : public ::testing::TestWithParam<CompressedParams> {
protected:
    std::pair<int, int> get_scale_dims(const CompressedParams& p, int oc) const {
        switch (p.scale_config) {
            case ScaleConfig::PerTensor: return {1, 1};
            case ScaleConfig::Grouped: return {p.scale_groups, oc};
            case ScaleConfig::PerOC: return {1, oc};
        }
        return {1, 1};
    }

    template <typename QType, typename ZpType>
    void run_8bit_test(engine& engine, const CompressedParams& p) {
        const int batch = p.batch, ifm = p.ifm, hidden = p.hidden;

        auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
        std::vector<float> src(batch * ifm);
        for (size_t i = 0; i < src.size(); i++)
            src[i] = 0.1f * static_cast<float>(static_cast<int>(i % 7) - 3);
        set_values(src_mem, to_f16(src));

        auto gate_mem = engine.allocate_memory({{ifm, hidden}, p.weight_dt, format::bfyx});
        auto up_mem = engine.allocate_memory({{ifm, hidden}, p.weight_dt, format::bfyx});
        auto down_mem = engine.allocate_memory({{hidden, ifm}, p.weight_dt, format::bfyx});

        std::vector<QType> gate_q(ifm * hidden), up_q(ifm * hidden), down_q(hidden * ifm);
        if constexpr (std::is_unsigned_v<QType>) {
            for (size_t i = 0; i < gate_q.size(); i++)
                gate_q[i] = static_cast<QType>(126 + (i % 7));
            for (size_t i = 0; i < up_q.size(); i++)
                up_q[i] = static_cast<QType>(124 + (i % 9));
            for (size_t i = 0; i < down_q.size(); i++)
                down_q[i] = static_cast<QType>(123 + (i % 11));
        } else {
            for (size_t i = 0; i < gate_q.size(); i++)
                gate_q[i] = static_cast<QType>(-3 + static_cast<int>(i % 7));
            for (size_t i = 0; i < up_q.size(); i++)
                up_q[i] = static_cast<QType>(-4 + static_cast<int>(i % 9));
            for (size_t i = 0; i < down_q.size(); i++)
                down_q[i] = static_cast<QType>(-5 + static_cast<int>(i % 11));
        }
        set_values(gate_mem, gate_q);
        set_values(up_mem, up_q);
        set_values(down_mem, down_q);

        auto [sr_gu, sc_gu] = get_scale_dims(p, hidden);
        auto [sr_d, sc_d] = get_scale_dims(p, ifm);

        auto s_gate_mem = engine.allocate_memory({{sr_gu, sc_gu}, data_types::f16, format::bfyx});
        auto s_up_mem = engine.allocate_memory({{sr_gu, sc_gu}, data_types::f16, format::bfyx});
        auto s_down_mem = engine.allocate_memory({{sr_d, sc_d}, data_types::f16, format::bfyx});

        int s_gu_count = sr_gu * sc_gu;
        int s_d_count = sr_d * sc_d;
        std::vector<float> s_gate(s_gu_count), s_up(s_gu_count), s_down(s_d_count);
        for (int i = 0; i < s_gu_count; i++) {
            s_gate[i] = 0.02f + 0.01f * i;
            s_up[i] = 0.025f + 0.01f * i;
        }
        for (int i = 0; i < s_d_count; i++)
            s_down[i] = 0.03f + 0.01f * i;
        set_values(s_gate_mem, to_f16(s_gate));
        set_values(s_up_mem, to_f16(s_up));
        set_values(s_down_mem, to_f16(s_down));

        auto zp_gate_mem = engine.allocate_memory({{sr_gu, sc_gu}, p.zp_dt, format::bfyx});
        auto zp_up_mem = engine.allocate_memory({{sr_gu, sc_gu}, p.zp_dt, format::bfyx});
        auto zp_down_mem = engine.allocate_memory({{sr_d, sc_d}, p.zp_dt, format::bfyx});

        std::vector<ZpType> zp_gate(s_gu_count), zp_up(s_gu_count), zp_down(s_d_count);
        if constexpr (std::is_unsigned_v<ZpType>) {
            for (int i = 0; i < s_gu_count; i++) {
                zp_gate[i] = static_cast<ZpType>(127 + (i % 3));
                zp_up[i] = static_cast<ZpType>(126 + (i % 3));
            }
            for (int i = 0; i < s_d_count; i++)
                zp_down[i] = static_cast<ZpType>(125 + (i % 3));
        } else {
            for (int i = 0; i < s_gu_count; i++) {
                zp_gate[i] = static_cast<ZpType>(i % 3);
                zp_up[i] = static_cast<ZpType>(1 + (i % 3));
            }
            for (int i = 0; i < s_d_count; i++)
                zp_down[i] = static_cast<ZpType>(-1 + (i % 3));
        }
        set_values(zp_gate_mem, zp_gate);
        set_values(zp_up_mem, zp_up);
        set_values(zp_down_mem, zp_down);

        auto gate = dequantize_weights<QType, ZpType>(gate_q, ifm, hidden, s_gate, sr_gu, sc_gu, zp_gate, sr_gu, sc_gu);
        auto up = dequantize_weights<QType, ZpType>(up_q, ifm, hidden, s_up, sr_gu, sc_gu, zp_up, sr_gu, sc_gu);
        auto down = dequantize_weights<QType, ZpType>(down_q, hidden, ifm, s_down, sr_d, sc_d, zp_down, sr_d, sc_d);
        auto ref = gated_mlp_reference(src, gate, up, down, batch, ifm, hidden);

        auto net = create_compressed_network(engine, batch, ifm, hidden,
                                             src_mem, gate_mem, up_mem, down_mem,
                                             s_gate_mem, s_up_mem, s_down_mem,
                                             zp_gate_mem, zp_up_mem, zp_down_mem);
        auto outputs = net->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "gmlp");
        check_output(outputs.at("gmlp").get_memory(), ref, p.tolerance);
    }

    void run_4bit_test(engine& engine, const CompressedParams& p) {
        const int batch = p.batch, ifm = p.ifm, hidden = p.hidden;

        auto src_mem = engine.allocate_memory({{batch, 1, 1, ifm}, data_types::f16, format::bfyx});
        std::vector<float> src(batch * ifm);
        for (size_t i = 0; i < src.size(); i++)
            src[i] = 0.1f * static_cast<float>(static_cast<int>(i % 7) - 3);
        set_values(src_mem, to_f16(src));

        auto gate_mem = engine.allocate_memory({{ifm, hidden}, p.weight_dt, format::bfyx});
        auto up_mem = engine.allocate_memory({{ifm, hidden}, p.weight_dt, format::bfyx});
        auto down_mem = engine.allocate_memory({{hidden, ifm}, p.weight_dt, format::bfyx});

        uint8_t gate_val = (p.weight_dt == data_types::u4) ? 0x77 : 0x11;
        uint8_t up_val = (p.weight_dt == data_types::u4) ? 0x66 : 0x22;
        uint8_t down_val = (p.weight_dt == data_types::u4) ? 0x55 : 0x33;

        std::vector<uint8_t> gate_q(ifm * hidden / 2, gate_val);
        std::vector<uint8_t> up_q(ifm * hidden / 2, up_val);
        std::vector<uint8_t> down_q(hidden * ifm / 2, down_val);
        set_values(gate_mem, gate_q);
        set_values(up_mem, up_q);
        set_values(down_mem, down_q);

        auto [sr_gu, sc_gu] = get_scale_dims(p, hidden);
        auto [sr_d, sc_d] = get_scale_dims(p, ifm);

        auto s_gate_mem = engine.allocate_memory({{sr_gu, sc_gu}, data_types::f16, format::bfyx});
        auto s_up_mem = engine.allocate_memory({{sr_gu, sc_gu}, data_types::f16, format::bfyx});
        auto s_down_mem = engine.allocate_memory({{sr_d, sc_d}, data_types::f16, format::bfyx});

        int s_gu_count = sr_gu * sc_gu;
        int s_d_count = sr_d * sc_d;
        std::vector<float> s_gate(s_gu_count), s_up(s_gu_count), s_down(s_d_count);
        if (p.zero_scale) {
            std::fill(s_gate.begin(), s_gate.end(), 0.0f);
            std::fill(s_up.begin(), s_up.end(), 0.0f);
            std::fill(s_down.begin(), s_down.end(), 0.0f);
        } else {
            for (int i = 0; i < s_gu_count; i++) {
                s_gate[i] = 0.02f + 0.01f * i;
                s_up[i] = 0.03f + 0.01f * i;
            }
            for (int i = 0; i < s_d_count; i++)
                s_down[i] = 0.04f + 0.01f * i;
        }
        set_values(s_gate_mem, to_f16(s_gate));
        set_values(s_up_mem, to_f16(s_up));
        set_values(s_down_mem, to_f16(s_down));

        bool is_unsigned_zp = (p.zp_dt == data_types::u8);
        auto zp_gate_mem = engine.allocate_memory({{sr_gu, sc_gu}, p.zp_dt, format::bfyx});
        auto zp_up_mem = engine.allocate_memory({{sr_gu, sc_gu}, p.zp_dt, format::bfyx});
        auto zp_down_mem = engine.allocate_memory({{sr_d, sc_d}, p.zp_dt, format::bfyx});

        if (is_unsigned_zp) {
            auto make_zp = [](int count, uint8_t base) {
                std::vector<uint8_t> zp(count);
                for (int i = 0; i < count; i++)
                    zp[i] = static_cast<uint8_t>(base + (i % 3));
                return zp;
            };
            set_values(zp_gate_mem, make_zp(s_gu_count, 7));
            set_values(zp_up_mem, make_zp(s_gu_count, 6));
            set_values(zp_down_mem, make_zp(s_d_count, 5));
        } else {
            auto make_zp = [](int count, int8_t base) {
                std::vector<int8_t> zp(count);
                for (int i = 0; i < count; i++)
                    zp[i] = static_cast<int8_t>(base + (i % 3));
                return zp;
            };
            set_values(zp_gate_mem, make_zp(s_gu_count, 1));
            set_values(zp_up_mem, make_zp(s_gu_count, 2));
            set_values(zp_down_mem, make_zp(s_d_count, 3));
        }

        auto net = create_compressed_network(engine, batch, ifm, hidden,
                                             src_mem, gate_mem, up_mem, down_mem,
                                             s_gate_mem, s_up_mem, s_down_mem,
                                             zp_gate_mem, zp_up_mem, zp_down_mem);
        auto outputs = net->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "gmlp");

        if (p.verify == VerifyMode::Zero) {
            std::vector<float> ref(batch * ifm, 0.0f);
            check_output(outputs.at("gmlp").get_memory(), ref, p.tolerance);
        } else {
            check_output_finite(outputs.at("gmlp").get_memory());
        }
    }
};

TEST_P(GatedMlpGPU_Compressed, basic) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "gated_mlp oneDNN implementation requires IMMAD support";
    }
    const auto& p = GetParam();

    if (p.weight_dt == data_types::u8) {
        run_8bit_test<uint8_t, uint8_t>(engine, p);
    } else if (p.weight_dt == data_types::i8) {
        run_8bit_test<int8_t, int8_t>(engine, p);
    } else {
        run_4bit_test(engine, p);
    }
}

INSTANTIATE_TEST_SUITE_P(
    gated_mlp_gpu,
    GatedMlpGPU_Compressed,
    ::testing::Values(
        // 8-bit with reference check
        CompressedParams{2, 4, 3, data_types::u8, data_types::u8, ScaleConfig::PerTensor, 1, false, 2e-2f, VerifyMode::Reference},
        CompressedParams{32, 32, 32, data_types::u8, data_types::u8, ScaleConfig::Grouped, 2, false, 1e-1f, VerifyMode::Reference},
        CompressedParams{1, 4, 3, data_types::u8, data_types::u8, ScaleConfig::PerOC, 1, false, 3e-2f, VerifyMode::Reference},
        CompressedParams{2, 4, 4, data_types::i8, data_types::i8, ScaleConfig::PerTensor, 1, false, 5e-2f, VerifyMode::Reference},
        // 4-bit zero-scale
        CompressedParams{2, 32, 32, data_types::u4, data_types::u8, ScaleConfig::PerTensor, 1, true, 1e-3f, VerifyMode::Zero},
        CompressedParams{2, 32, 32, data_types::i4, data_types::i8, ScaleConfig::PerTensor, 1, true, 1e-3f, VerifyMode::Zero},
        // 4-bit grouped
        CompressedParams{8, 48, 48, data_types::u4, data_types::u8, ScaleConfig::Grouped, 2, false, 0.0f, VerifyMode::Finite},
        CompressedParams{8, 48, 48, data_types::i4, data_types::i8, ScaleConfig::Grouped, 2, false, 0.0f, VerifyMode::Finite}
    ),
    [](const ::testing::TestParamInfo<CompressedParams>& info) {
        const auto& p = info.param;
        std::string dt_name;
        switch (p.weight_dt) {
            case data_types::u8: dt_name = "u8"; break;
            case data_types::i8: dt_name = "i8"; break;
            case data_types::u4: dt_name = "u4"; break;
            case data_types::i4: dt_name = "i4"; break;
            default: dt_name = "unknown"; break;
        }
        std::string sc_name;
        switch (p.scale_config) {
            case ScaleConfig::PerTensor: sc_name = "per_tensor"; break;
            case ScaleConfig::Grouped: sc_name = "grouped"; break;
            case ScaleConfig::PerOC: sc_name = "per_oc"; break;
        }
        std::string name = dt_name + "_" + sc_name;
        if (p.zero_scale) name += "_zero_scale";
        name += "_batch" + std::to_string(p.batch);
        return name;
    }
);

#endif
