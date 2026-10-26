// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include "fully_connected_inst.h"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

enum class ZpKind { None, Scalar, Grouped };

struct Int3FcParams {
    // An empty seq_len gives a 2D [batch, K] input, otherwise the input is [batch, seq_len, K].
    long batch;
    long seq_len;
    long ifm;
    long ofm;
    long scale_group;
    ZpKind zp;
    bool bias;
    bool fused_eltwise;
    // Scale and zero point in fbyx, as constant transposes leave them.
    bool transposed_decompression = false;
    // With fused_eltwise: swish of the FC times the eltwise input (the gate of a SwiGLU) instead of their sum.
    bool swish_mul = false;
};

class fully_connected_int3_gpu_tests : public ::testing::Test {
public:
    void SetUp() override {
        if (get_test_engine().get_device_info().arch < gpu_arch::xe2) {
            GTEST_SKIP();
        }
    }

    // Runs the FC on u3 weights and compares it with the reference kernel. For every row count in rows_list
    // (batch for a 2D input, seq_len for a 3D one), expected_gemms lists the GEMM variants a static impl has to hold.
    // A dynamic network runs every new row count on one dynamic impl holding all variants. With
    // wait_for_static_impls it reruns each row count on the static impl compiled asynchronously for it.
    void run(const Int3FcParams& p,
             bool is_dynamic,
             bool is_caching_test,
             const std::vector<long>& rows_list,
             const std::vector<std::vector<std::string>>& expected_gemms = {},
             bool wait_for_static_impls = false) {
        ASSERT_TRUE(is_dynamic || rows_list.size() == 1);
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();
        const bool is_3d = p.seq_len != 0;

        auto make_input_ps = [&](long rows) {
            return is_3d ? ov::PartialShape{p.batch, rows, p.ifm} : ov::PartialShape{rows, p.ifm};
        };
        const auto dyn_input_ps = is_3d ? ov::PartialShape{-1, -1, p.ifm} : ov::PartialShape{-1, p.ifm};
        const auto bias_ps = is_3d ? ov::PartialShape{1, 1, p.ofm} : ov::PartialShape{1, p.ofm};
        const long groups = p.ifm / p.scale_group;
        const auto decompression_format = p.transposed_decompression ? format::fbyx : format::bfyx;

        const auto weights_vals = rg.generate_random_1d<uint8_t>(p.ofm * p.ifm * 3 / 8, 0, 255);
        const auto scale_vals = rg.generate_random_1d<ov::float16>(p.ofm * groups, -0.05f, 0.05f, 1024);
        const auto zp_vals = rg.generate_random_1d<uint8_t>(p.ofm * groups, 2, 5);
        const auto bias_vals = rg.generate_random_1d<ov::float16>(p.ofm, -1.0f, 1.0f);
        const std::string zp_id = p.zp == ZpKind::Grouped ? "zp" : "";
        const std::string bias_id = p.bias ? "bias" : "";
        const std::string last = p.fused_eltwise ? "eltw" : "fc_prim";

        // Each network gets its own constants: building a program on a discrete GPU moves them to device memory.
        auto make_topology = [&]() {
            auto weights_mem = engine.allocate_memory({{p.ofm, p.ifm}, data_types::u3, format::bfyx});
            set_values(weights_mem, weights_vals);
            auto scale_mem = engine.allocate_memory({{p.ofm, groups}, data_types::f16, decompression_format});
            set_values(scale_mem, scale_vals);

            topology topology(data("weights", weights_mem), data("scale", scale_mem));
            if (p.zp == ZpKind::Grouped) {
                auto zp_mem = engine.allocate_memory({{p.ofm, groups}, data_types::u8, decompression_format});
                set_values(zp_mem, zp_vals);
                topology.add(data("zp", zp_mem));
            }
            if (p.bias) {
                auto bias_mem = engine.allocate_memory({bias_ps, data_types::f16, format::bfyx});
                set_values(bias_mem, bias_vals);
                topology.add(data("bias", bias_mem));
            }
            auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", bias_id, "scale", zp_id, data_types::f16, is_3d ? 3 : 2, 2);
            if (p.zp == ZpKind::Scalar) {
                fc_prim.decompression_zero_point_scalar = 4.0f;
            }
            topology.add(fc_prim);
            if (p.fused_eltwise) {
                topology.add(input_layout("eltw_data",
                                          layout{is_dynamic ? (is_3d ? ov::PartialShape{-1, -1, p.ofm} : ov::PartialShape{-1, p.ofm})
                                                            : (is_3d ? ov::PartialShape{p.batch, rows_list[0], p.ofm} : ov::PartialShape{rows_list[0], p.ofm}),
                                                 data_types::f16,
                                                 format::bfyx}));
                if (p.swish_mul) {
                    topology.add(activation("swish", input_info("fc_prim"), activation_func::swish, {1.0f, 0.0f}));
                    topology.add(eltwise("eltw", {input_info("swish"), input_info("eltw_data")}, eltwise_mode::prod));
                } else {
                    topology.add(eltwise("eltw", {input_info("fc_prim"), input_info("eltw_data")}, eltwise_mode::sum));
                }
            } else {
                // A second user keeps the output reorder from being fused, so the FC stays internal and gets
                // fake-aligned row counts as in a real model.
                topology.add(reorder("output2", input_info("fc_prim"), format::bfyx, data_types::f32));
            }
            topology.add(reorder("output", input_info(last), format::bfyx, data_types::f32));

            topology.add(input_layout("input", layout{is_dynamic ? dyn_input_ps : make_input_ps(rows_list[0]), data_types::f16, format::bfyx}));
            return topology;
        };

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(128));
        auto topology = make_topology();
        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        auto ref_config = get_test_default_config(engine);
        ref_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        ref_config.set_property(ov::intel_gpu::optimize_data(true));
        ov::intel_gpu::ImplementationDesc ref_impl = {format::bfyx, "fully_connected_gpu_bfyx_ref", impl_types::ocl};
        ref_config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", ref_impl}}));
        network ref_net(engine, make_topology(), ref_config);

        const primitive_impl* first_impl = nullptr;
        for (size_t i = 0; i < rows_list.size(); ++i) {
            const long rows = rows_list[i];
            const auto input_ps = make_input_ps(rows);
            auto input_mem = engine.allocate_memory({input_ps, data_types::f16, format::bfyx});
            set_values(input_mem, rg.generate_random_1d<ov::float16>(ov::shape_size(input_ps.to_shape()), -2.0f, 2.0f));
            net->set_input_data("input", input_mem);
            ref_net.set_input_data("input", input_mem);
            if (p.fused_eltwise) {
                auto eltw_ps = input_ps;
                eltw_ps[eltw_ps.size() - 1] = p.ofm;
                auto eltw_mem = engine.allocate_memory({eltw_ps, data_types::f16, format::bfyx});
                set_values(eltw_mem, rg.generate_random_1d<ov::float16>(ov::shape_size(eltw_ps.to_shape()), -1.0f, 1.0f));
                net->set_input_data("eltw_data", eltw_mem);
                ref_net.set_input_data("eltw_data", eltw_mem);
            }

            auto outputs = net->execute();
            if (wait_for_static_impls) {
                net->get_program()->get_compilation_context().wait_all();
                outputs = net->execute();
            }
            auto ref_outputs = ref_net.execute();

            auto inst = net->get_primitive("fc_prim");
            auto impl = inst->get_impl();
            ASSERT_NE(impl, nullptr);
            // Async compilation may swap in a static impl for a row count the dynamic network has seen before.
            if (!is_dynamic || wait_for_static_impls) {
                ASSERT_FALSE(impl->is_dynamic()) << "rows = " << rows;
            }
            // A static impl of a dynamic model takes the quantizer from the dynamic impl, and a single-row one its GEMV too
            if (wait_for_static_impls) {
                const size_t total_rows = is_3d ? p.batch * rows : rows;
                ASSERT_EQ(impl->get_kernels().size(), total_rows == 1 ? 0u : 1u) << "rows = " << rows;
            }
            ASSERT_EQ(impl->get_kernel_name().rfind("ocl::fc_int3_dpas", 0), 0u) << "rows = " << rows << ", impl = " << impl->get_kernel_name();
            if (impl->is_dynamic()) {
                if (first_impl == nullptr) {
                    first_impl = impl;
                }
                ASSERT_EQ(impl, first_impl) << "rows = " << rows;
            }
            // Static f16 FCs do not take fused eltwise ops (fc_supports_fusings).
            if (p.fused_eltwise && is_dynamic) {
                ASSERT_TRUE(inst->has_fused_primitives()) << "rows = " << rows;
            }
            std::vector<std::string> gemms;
            for (const auto& kernel : impl->get_kernels()) {
                const auto& id = kernel->get_id();
                if (id.find("fully_connected_int3_dpas_quantize_") == std::string::npos) {
                    gemms.push_back(id);
                }
            }
            if (impl->is_dynamic()) {
                const auto variants = is_xe3() ? std::vector<std::string>{"scalar", "v1_t32_sg1"}
                                               : std::vector<std::string>{"scalar", "v1_t8", "v1_t16", "v1_t32_sg1"};
                for (const auto& variant : variants) {
                    ASSERT_TRUE(std::any_of(gemms.begin(),
                                            gemms.end(),
                                            [&](const std::string& id) {
                                                return id.find("fully_connected_int3_dpas_" + variant + "_") != std::string::npos;
                                            }))
                        << "missing " << variant;
                }
            } else if (i < expected_gemms.size() && !expected_gemms[i].empty()) {
                ASSERT_EQ(gemms.size(), expected_gemms[i].size()) << "rows = " << rows;
                for (size_t k = 0; k < gemms.size(); ++k) {
                    ASSERT_NE(gemms[k].find("fully_connected_int3_dpas_" + expected_gemms[i][k] + "_"), std::string::npos)
                        << "rows = " << rows << ", kernel = " << gemms[k];
                }
            }

            const size_t count = outputs.at("output").get_layout().count();
            ASSERT_EQ(count, ref_outputs.at("output").get_layout().count());
            cldnn::mem_lock<float> out(outputs.at("output").get_memory(), get_test_stream());
            cldnn::mem_lock<float> ref(ref_outputs.at("output").get_memory(), get_test_stream());

            // Activations are quantized to int8 per group, so compare against the output magnitude.
            float max_ref = 0.f;
            double err2 = 0.0;
            double ref2 = 0.0;
            for (size_t j = 0; j < count; ++j) {
                max_ref = std::max(max_ref, std::abs(ref[j]));
                err2 += (out[j] - ref[j]) * (out[j] - ref[j]);
                ref2 += ref[j] * ref[j];
            }
            for (size_t j = 0; j < count; ++j) {
                ASSERT_NEAR(out[j], ref[j], 0.05f * max_ref) << "rows = " << rows << ", j = " << j;
            }
            ASSERT_LT(std::sqrt(err2 / std::max(ref2, 1e-12)), 0.02) << "rows = " << rows;
        }
        ASSERT_EQ(first_impl != nullptr, is_dynamic && !wait_for_static_impls);
    }

    static bool is_xe3() {
        return get_test_engine().get_device_info().arch >= gpu_arch::xe3;
    }

    static bool is_igpu() {
        return get_test_engine().get_device_info().dev_type == device_type::integrated_gpu;
    }
};

const Int3FcParams base{0, 0, 1024, 1024, 128, ZpKind::Scalar, false, false};

Int3FcParams with(Int3FcParams p, long batch, long seq_len = 0) {
    p.batch = batch;
    p.seq_len = seq_len;
    return p;
}

}  // namespace

TEST_F(fully_connected_int3_gpu_tests, scalar_gemm_single_row) {
    run(base, false, false, {1}, {{"scalar"}});
}

TEST_F(fully_connected_int3_gpu_tests, v1_t8_few_rows) {
    run(base, false, false, {2}, {{is_xe3() ? "v1_t32_sg1" : "v1_t8"}});
}

TEST_F(fully_connected_int3_gpu_tests, v1_t8) {
    run(base, false, false, {8}, {{is_xe3() ? "v1_t32_sg1" : "v1_t8"}});
}

TEST_F(fully_connected_int3_gpu_tests, v1_t16) {
    run(base, false, false, {16}, {{is_xe3() ? "v1_t32_sg1" : "v1_t16"}});
}

TEST_F(fully_connected_int3_gpu_tests, v1_t32) {
    run(base, false, false, {64});
}

TEST_F(fully_connected_int3_gpu_tests, v1_grouped_zp) {
    auto p = base;
    p.zp = ZpKind::Grouped;
    run(p, false, false, {128}, {{"v1_t32_sg4"}});
}

TEST_F(fully_connected_int3_gpu_tests, dynamic_no_zp) {
    auto p = base;
    p.zp = ZpKind::None;
    run(p, true, false, {3, 32});
}

TEST_F(fully_connected_int3_gpu_tests, v2_sg8) {
    run(base, false, false, {128}, {{"v2_sg8"}});
}

TEST_F(fully_connected_int3_gpu_tests, v2_sg16) {
    auto p = base;
    p.ofm = 2048;
    run(p, false, false, {256}, {{"v2_sg16"}});
}

TEST_F(fully_connected_int3_gpu_tests, dynamic_scale_group_64) {
    auto p = base;
    p.scale_group = 64;
    p.zp = ZpKind::Grouped;
    run(p, true, false, {1, 40});
}

TEST_F(fully_connected_int3_gpu_tests, transposed_scale_zp) {
    auto p = base;
    p.zp = ZpKind::Grouped;
    p.transposed_decompression = true;
    run(p, true, false, {1, 48});
}

TEST_F(fully_connected_int3_gpu_tests, transposed_scale_v2) {
    auto p = base;
    p.transposed_decompression = true;
    run(p, false, false, {128}, {{"v2_sg8"}});
}

TEST_F(fully_connected_int3_gpu_tests, bias_and_eltwise) {
    auto p = base;
    p.bias = true;
    p.fused_eltwise = true;
    run(p, false, false, {24});
}

TEST_F(fully_connected_int3_gpu_tests, dynamic_bias_and_fused_eltwise) {
    auto p = base;
    p.bias = true;
    p.fused_eltwise = true;
    run(p, true, false, {24, 3, 130});
}

TEST_F(fully_connected_int3_gpu_tests, dynamic_3d_batch_2_fused_eltwise) {
    auto p = with(base, 2, 1);
    p.fused_eltwise = true;
    run(p, true, false, {40, 1, 100});
}

TEST_F(fully_connected_int3_gpu_tests, dynamic_2d) {
    auto p = base;
    p.bias = true;
    run(p, true, false, {1, 5, 8, 100, 300, 7, 1});
}

TEST_F(fully_connected_int3_gpu_tests, dynamic_3d) {
    run(with(base, 2, 1), true, false, {1, 40, 3});
}

TEST_F(fully_connected_int3_gpu_tests, dynamic_3d_fused_eltwise) {
    auto p = with(base, 1, 1);
    p.fused_eltwise = true;
    run(p, true, false, {1, 17, 130, 2});
}

TEST_F(fully_connected_int3_gpu_tests, dynamic_grouped_zp) {
    auto p = base;
    p.zp = ZpKind::Grouped;
    run(p, true, false, {1, 33, 4});
}

TEST_F(fully_connected_int3_gpu_tests, dynamic_static_impls_reuse_stages) {
    auto p = base;
    p.bias = true;
    run(p, true, false, {1, 40, 1}, {}, true);
}

TEST_F(fully_connected_int3_gpu_tests, dynamic_3d_fused_eltwise_static_impls_reuse_stages) {
    auto p = with(base, 1, 1);
    p.fused_eltwise = true;
    run(p, true, false, {1, 20, 1}, {}, true);
}

// Static v2 impls read a fused eltwise input shaped like the output as 2D blocks; the row counts leave a partial
// last tile.
TEST_F(fully_connected_int3_gpu_tests, dynamic_3d_fused_eltwise_static_impls_v2) {
    auto p = with(base, 1, 1);
    p.fused_eltwise = true;
    run(p, true, false, {40, 130, 300}, {}, true);
}

// The gate of a SwiGLU: the v2 epilogue computes the fused swish itself, ahead of the 2D block eltwise input.
TEST_F(fully_connected_int3_gpu_tests, dynamic_3d_fused_swish_mul_static_impls_v2) {
    auto p = with(base, 1, 1);
    p.fused_eltwise = true;
    p.swish_mul = true;
    run(p, true, false, {40, 130}, {}, true);
}

TEST_F(fully_connected_int3_gpu_tests, dynamic_fused_eltwise_static_impls_v2_k_above_n) {
    auto p = base;
    p.ifm = 2048;
    p.bias = true;
    p.fused_eltwise = true;
    run(p, true, false, {100, 211}, {}, true);
}

TEST_F(fully_connected_int3_gpu_tests, cached_static) {
    run(base, false, true, {1}, {{"scalar"}});
}

TEST_F(fully_connected_int3_gpu_tests, cached_static_v2) {
    run(base, false, true, {128}, {{"v2_sg8"}});
}

TEST_F(fully_connected_int3_gpu_tests, cached_dynamic) {
    auto p = with(base, 1, 1);
    p.fused_eltwise = true;
    run(p, true, true, {1, 20, 3});
}

TEST_F(fully_connected_int3_gpu_tests, cached_dynamic_static_impls_reuse_stages) {
    auto p = with(base, 1, 1);
    p.fused_eltwise = true;
    run(p, true, true, {1, 20, 1}, {}, true);
}
