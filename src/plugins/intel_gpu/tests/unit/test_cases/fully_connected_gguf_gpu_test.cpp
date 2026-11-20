// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected_gguf.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <random>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gguf_dequantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

using ov::op::internal::GGUFDequantize;
using Format = GGUFDequantize::Format;

void put_f16(uint8_t* p, float v) {
    const uint16_t bits = ov::float16(v).to_bits();
    std::memcpy(p, &bits, sizeof(bits));
}

// Random quant bits and scales, with small positive f16 super-scales (Q5_K: d and dmin at byte 0, Q6_K: d at byte 208).
std::vector<uint8_t> random_blocks(Format format, size_t n, size_t k, std::mt19937& rng) {
    const size_t bb = GGUFDequantize::block_bytes(format);
    const size_t nblk = n * k / GGUFDequantize::block_size;
    std::vector<uint8_t> bytes(nblk * bb);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.0005f, 0.004f);
    for (auto& b : bytes) {
        b = static_cast<uint8_t>(byte(rng));
    }
    for (size_t i = 0; i < nblk; ++i) {
        uint8_t* blk = bytes.data() + i * bb;
        if (format == Format::Q5_K) {
            put_f16(blk, scale(rng));
            put_f16(blk + 2, scale(rng));
        } else {
            put_f16(blk + 208, scale(rng));
        }
    }
    return bytes;
}

// Q6_K weights as ConvertMatMulToFullyConnectedGGUF passes them: QL [nblk][128] | QH [nblk][64] | SC [nblk][16] | D [nblk].
std::vector<uint8_t> q6_k_planes(const std::vector<uint8_t>& blocks) {
    const size_t bb = GGUFDequantize::block_bytes(Format::Q6_K);
    const size_t nblk = blocks.size() / bb;
    std::vector<uint8_t> planes(blocks.size());
    for (size_t i = 0; i < nblk; ++i) {
        const uint8_t* b = blocks.data() + i * bb;
        std::memcpy(planes.data() + i * 128, b, 128);
        std::memcpy(planes.data() + nblk * 128 + i * 64, b + 128, 64);
        std::memcpy(planes.data() + nblk * 192 + i * 16, b + 192, 16);
        std::memcpy(planes.data() + nblk * 208 + i * 2, b + 208, 2);
    }
    return planes;
}

struct Reference {
    std::vector<float> y;      // x * W^T
    std::vector<float> scale;  // sum |x * w| per output, the magnitude the error is measured against
};

Reference reference(Format format, const std::vector<uint8_t>& blocks, const std::vector<ov::float16>& x, size_t m, size_t n, size_t k) {
    // W from GGUFDequantize::evaluate
    auto data = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{n, blocks.size() / n}, blocks.data());
    GGUFDequantize dequantize(data, format);
    ov::TensorVector w{ov::Tensor(ov::element::f32, ov::Shape{n, k})};
    OPENVINO_ASSERT(dequantize.evaluate(w, {data->get_tensor_view()}));
    const auto* wp = w[0].data<const float>();

    Reference r{std::vector<float>(m * n), std::vector<float>(m * n)};
    for (size_t t = 0; t < m; ++t) {
        for (size_t o = 0; o < n; ++o) {
            double acc = 0.0, mag = 0.0;
            for (size_t i = 0; i < k; ++i) {
                const double p = static_cast<double>(wp[o * k + i]) * static_cast<float>(x[t * k + i]);
                acc += p;
                mag += std::fabs(p);
            }
            r.y[t * n + o] = static_cast<float>(acc);
            r.scale[t * n + o] = static_cast<float>(mag);
        }
    }
    return r;
}

struct TestCase {
    Format format;
    size_t n;
    size_t k;
    std::vector<size_t> row_counts;  // executed in this order on one network
    bool dynamic;
    bool input_3d;  // [1, M, K] instead of [M, K]
    bool is_caching_test;
    std::string prefill_path = "";  // ov::intel_gpu::gguf_prefill_path
};

void run_test(const TestCase& tc) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_imad) {
        GTEST_SKIP();
    }
    const size_t n = tc.n, k = tc.k;
    std::mt19937 rng(static_cast<uint32_t>(n * 131 + k));
    const auto blocks = random_blocks(tc.format, n, k, rng);

    auto weights = engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(n), static_cast<int64_t>(blocks.size() / n)}, data_types::u8, format::bfyx});
    set_values(weights, tc.format == Format::Q6_K ? q6_k_planes(blocks) : blocks);

    auto shape_of = [&](int64_t m) {
        return tc.input_3d ? ov::PartialShape{1, m, static_cast<int64_t>(k)} : ov::PartialShape{m, static_cast<int64_t>(k)};
    };
    const auto in_shape = tc.dynamic ? shape_of(-1) : shape_of(static_cast<int64_t>(tc.row_counts[0]));
    topology topology(input_layout("input", layout{in_shape, data_types::f16, format::bfyx}),
                      data("weights", weights),
                      fully_connected_gguf("fc", input_info("input"), input_info("weights"), tc.format, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::gguf_prefill_path(tc.prefill_path));
    auto network = get_network(engine, topology, config, get_test_stream_ptr(), tc.is_caching_test);

    std::uniform_real_distribution<float> val(-1.0f, 1.0f);
    for (const auto m : tc.row_counts) {
        std::vector<ov::float16> x(m * k);
        for (auto& v : x) {
            v = ov::float16(val(rng));
        }
        auto input = engine.allocate_memory({shape_of(static_cast<int64_t>(m)), data_types::f16, format::bfyx});
        set_values(input, x);
        network->set_input_data("input", input);
        auto outputs = network->execute();
        auto expected_shape = shape_of(static_cast<int64_t>(m)).to_shape();
        expected_shape.back() = n;
        ASSERT_EQ(outputs.at("fc").get_layout().get_shape(), expected_shape);
        auto output = outputs.at("fc").get_memory();

        const auto ref = reference(tc.format, blocks, x, m, n, k);
        cldnn::mem_lock<ov::float16, mem_lock_type::read> y(output, get_test_stream());
        // int8 activations stay well within 2% of the summed magnitudes; a layout error is off by the order of the result.
        for (size_t i = 0; i < m * n; ++i) {
            ASSERT_NEAR(static_cast<float>(y[i]), ref.y[i], 0.02f * ref.scale[i] + 1e-3f) << "M=" << m << " N=" << n << " K=" << k << " output " << i;
        }
    }
}

}  // namespace

TEST(fully_connected_gguf_gpu_test, q6_k_one_row) {
    run_test({Format::Q6_K, 64, 512, {1}, false, false, false});
}

TEST(fully_connected_gguf_gpu_test, q6_k_rows_not_multiple_of_four) {
    run_test({Format::Q6_K, 7, 256, {3}, false, false, false});
    run_test({Format::Q6_K, 130, 2560, {1}, false, false, false});
}

TEST(fully_connected_gguf_gpu_test, q6_k_many_rows) {
    run_test({Format::Q6_K, 96, 1024, {37}, false, false, false});
}

TEST(fully_connected_gguf_gpu_test, q6_k_3d_input) {
    run_test({Format::Q6_K, 64, 512, {5}, false, true, false});
}

TEST(fully_connected_gguf_gpu_test, q6_k_dynamic_rows) {
    run_test({Format::Q6_K, 130, 1024, {1, 40, 1, 7, 8, 1}, true, false, false});
}

// Every prefill tile (M <= 64, <= 128, <= 256, larger) and the switches between them.
TEST(fully_connected_gguf_gpu_test, q6_k_dynamic_rows_prefill_tiles) {
    run_test({Format::Q6_K, 130, 1024, {40, 100, 200, 300, 64, 129, 256, 1}, true, false, false});
}

TEST(fully_connected_gguf_gpu_test, q6_k_scratch_prefill) {
    run_test({Format::Q6_K, 130, 1024, {1, 40, 300, 1}, true, false, false, "scratch"});
}

TEST(fully_connected_gguf_gpu_test, q6_k_dynamic_rows_3d) {
    run_test({Format::Q6_K, 64, 512, {1, 9, 1}, true, true, false});
}

TEST(fully_connected_gguf_gpu_test, q6_k_cached) {
    run_test({Format::Q6_K, 64, 512, {1, 9, 1}, true, false, true});
}

TEST(fully_connected_gguf_gpu_test, q5_k_one_row) {
    run_test({Format::Q5_K, 64, 512, {1}, false, false, false});
}

TEST(fully_connected_gguf_gpu_test, q5_k_rows_not_multiple_of_four) {
    run_test({Format::Q5_K, 7, 256, {3}, false, false, false});
    run_test({Format::Q5_K, 130, 2560, {1}, false, false, false});
}

TEST(fully_connected_gguf_gpu_test, q5_k_many_rows) {
    run_test({Format::Q5_K, 96, 1024, {37}, false, false, false});
}

TEST(fully_connected_gguf_gpu_test, q5_k_3d_input) {
    run_test({Format::Q5_K, 64, 512, {5}, false, true, false});
}

TEST(fully_connected_gguf_gpu_test, q5_k_dynamic_rows) {
    run_test({Format::Q5_K, 130, 1024, {1, 40, 1, 7, 8, 1}, true, false, false});
}

TEST(fully_connected_gguf_gpu_test, q5_k_dynamic_rows_prefill_tiles) {
    run_test({Format::Q5_K, 130, 1024, {40, 100, 200, 300, 64, 129, 256, 1}, true, false, false});
}

TEST(fully_connected_gguf_gpu_test, q5_k_scratch_prefill) {
    run_test({Format::Q5_K, 130, 1024, {1, 40, 300, 1}, true, false, false, "scratch"});
}

TEST(fully_connected_gguf_gpu_test, q5_k_dynamic_rows_3d) {
    run_test({Format::Q5_K, 64, 512, {1, 9, 1}, true, true, false});
}

TEST(fully_connected_gguf_gpu_test, q5_k_cached) {
    run_test({Format::Q5_K, 64, 512, {1, 9, 1}, true, false, true});
}

namespace {

// A model with only static shapes takes the legacy shape infer unless an op requires the new one; the legacy one would
// make the [N, bytes] weights 4D. With residual, y = x * W^T + r and the Add must be fused into FullyConnectedGGUF.
void run_static_model_test(Format format, size_t m = 3, bool residual = false) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_imad) {
        GTEST_SKIP();
    }
    constexpr size_t n = 16, k = 256;
    std::mt19937 rng(7);
    const auto bytes = random_blocks(format, n, k, rng);
    auto blocks = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{n, bytes.size() / n}, bytes.data());
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, m, k});
    auto deq = std::make_shared<GGUFDequantize>(blocks, format);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(x, deq, false, true);
    auto r = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, m, n});
    x->output(0).get_tensor().set_names({"x"});
    r->output(0).get_tensor().set_names({"r"});
    auto add = std::make_shared<ov::op::v1::Add>(matmul, r);
    add->set_friendly_name("residual_add");
    auto model = residual ? std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{x, r})
                          : std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{x});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU", ov::hint::inference_precision(ov::element::f16), ov::hint::dynamic_quantization_group_size(32));
    auto request = compiled.create_infer_request();
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);
    std::vector<ov::float16> x16(m * k);
    ov::Tensor input(ov::element::f32, {1, 1, m, k});
    for (size_t i = 0; i < m * k; ++i) {
        x16[i] = ov::float16(val(rng));
        input.data<float>()[i] = static_cast<float>(x16[i]);
    }
    std::vector<ov::float16> r16(m * n);
    if (residual) {
        ov::Tensor res(ov::element::f32, {1, 1, m, n});
        for (size_t i = 0; i < m * n; ++i) {
            r16[i] = ov::float16(val(rng));
            res.data<float>()[i] = static_cast<float>(r16[i]);
        }
        request.set_tensor("r", res);
        bool fused = false;
        for (const auto& op : compiled.get_runtime_model()->get_ordered_ops()) {
            const auto& info = op->get_rt_info();
            const auto names = info.find(ov::exec_model_info::ORIGINAL_NAMES);
            const auto type = info.find(ov::exec_model_info::LAYER_TYPE);
            if (names != info.end() && type != info.end() && names->second.as<std::string>().find("residual_add") != std::string::npos) {
                auto t = type->second.as<std::string>();
                std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                fused |= t.find("gguf") != std::string::npos;
            }
        }
        ASSERT_TRUE(fused) << "residual Add not fused into FullyConnectedGGUF";
    }
    request.set_tensor("x", input);
    request.infer();
    const auto output = request.get_output_tensor();
    ASSERT_EQ(output.get_shape(), (ov::Shape{1, 1, m, n}));
    const auto ref = reference(format, bytes, x16, m, n, k);
    for (size_t i = 0; i < m * n; ++i) {
        ASSERT_NEAR(output.data<const float>()[i], ref.y[i] + static_cast<float>(r16[i]), 0.02f * ref.scale[i] + 2e-3f) << "M=" << m << " output " << i;
    }
}

}  // namespace

TEST(fully_connected_gguf_gpu_test, q6_k_static_model) {
    run_static_model_test(Format::Q6_K);
}

TEST(fully_connected_gguf_gpu_test, q5_k_static_model) {
    run_static_model_test(Format::Q5_K);
}

// M = 3: decode kernel, M = 37: prefill kernel
TEST(fully_connected_gguf_gpu_test, q6_k_fused_residual_add) {
    run_static_model_test(Format::Q6_K, 3, true);
    run_static_model_test(Format::Q6_K, 37, true);
}

TEST(fully_connected_gguf_gpu_test, q5_k_fused_residual_add) {
    run_static_model_test(Format::Q5_K, 3, true);
    run_static_model_test(Format::Q5_K, 37, true);
}
