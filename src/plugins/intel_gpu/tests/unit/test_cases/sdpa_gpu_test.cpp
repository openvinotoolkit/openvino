// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/runtime/debug_configuration.hpp>

#include "impls/ocl_v2/sdpa/sdpa_opt.hpp"
#include "openvino/util/file_util.hpp"
#include "program_wrapper.h"
#include <array>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scaled_dot_product_attention.hpp>
#include "scaled_dot_product_attention_inst.h"

#include <cstddef>
#include <vector>

using namespace cldnn;
using namespace ::tests;
// #define ENABLE_ONEDNN_FOR_GPU
namespace  {
#ifdef ENABLE_ONEDNN_FOR_GPU
struct sdpa_test_params {
    int head_size;
    int num_heads;
    int kv_num_heads;
    int sequence_length_q;
    int sequence_length_kv;
    int batch;
    bool dynamic;
    bool use_scalar_scale_val;
    float scale_val;
    bool use_scalar_attn_mask;
    float attn_mask_val;
    int bit_width;
    bool asymmetric;

    // Constructor for basic tests (backward compatibility)
    sdpa_test_params(int h_size, int n_heads, int seq_q, int seq_kv, int b,
                     bool dynamic_shape)
          : head_size(h_size), num_heads(n_heads), kv_num_heads(n_heads), sequence_length_q(seq_q),
          sequence_length_kv(seq_kv), batch(b), dynamic(dynamic_shape),
          use_scalar_scale_val(false), scale_val(1.0f), use_scalar_attn_mask(false),
            attn_mask_val(0.0f), bit_width(0), asymmetric(false) {}

    // Constructor for advanced caching tests
    sdpa_test_params(int h_size, int n_heads, int seq_q, int seq_kv, int b,
                     bool use_scale, float scale, bool use_mask, float mask)
          : head_size(h_size), num_heads(n_heads), kv_num_heads(n_heads), sequence_length_q(seq_q), sequence_length_kv(seq_kv),
          batch(b), dynamic(true), use_scalar_scale_val(use_scale),
            scale_val(scale), use_scalar_attn_mask(use_mask), attn_mask_val(mask), bit_width(0), asymmetric(false) {}
    
    // constructor for quantization tests
    sdpa_test_params(int h_size, int q_heads, int kv_heads, int seq_q, int seq_kv, int b, int bits, bool asym)
          : head_size(h_size), num_heads(q_heads), kv_num_heads(kv_heads), sequence_length_q(seq_q),
            sequence_length_kv(seq_kv), batch(b), dynamic(false), use_scalar_scale_val(false), scale_val(1.0f),
            use_scalar_attn_mask(false), attn_mask_val(0.0f), bit_width(bits), asymmetric(asym) {}
};

struct sdpa_gpu_test : public ::testing::TestWithParam<sdpa_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void load_input(cldnn::memory::ptr mem, size_t idx) {
        auto shapes = mem->get_layout().get_shape();
        size_t size = ov::shape_size(shapes);
        auto input_data = rg.generate_random_1d<ov::float16>(size, -1.0f, 1.0f);
        set_values(mem, input_data);
    }

    std::tuple<cldnn::memory::ptr, cldnn::network::ptr> run_network(bool is_caching_test, bool use_optimized_sdpa,
            cldnn::layout input0_layout,
            cldnn::layout input1_layout,
            cldnn::layout input2_layout,
            cldnn::layout input3_layout,
            cldnn::memory::ptr input0,
            cldnn::memory::ptr input1,
            cldnn::memory::ptr input2,
            cldnn::memory::ptr input3,
            bool use_scalar_scale_val = false,
            float scale_val = 1.0f,
            bool use_scalar_attn_mask = false,
            float attn_mask_val = 0.0f) {
        auto& engine = get_test_engine();
        topology topo;
        topo.add(input_layout("input0", input0_layout));
        topo.add(input_layout("input1", input1_layout));
        topo.add(input_layout("input2", input2_layout));
        topo.add(input_layout("input3", input3_layout));

        auto sdpa_prim = scaled_dot_product_attention("sdpa", {input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3")},
            false, -1, {0,2,1,3}, {0,2,1,3}, {0,2,1,3}, {0,1,2,3}, {}, false);

        if (use_scalar_scale_val) {
            sdpa_prim.scale_val = scale_val;
        }

        if (use_scalar_attn_mask) {
            sdpa_prim.attn_mask_val = attn_mask_val;
        }

        topo.add(sdpa_prim);
        topo.add(reorder("result",input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        if (use_optimized_sdpa) {
            if (!is_caching_test) {
                if (engine.get_device_info().supports_immad) {
                    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                        {"sdpa", {format::type::bfyx, "sdpa_micro"}} }));
                } else {
                    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                        {"sdpa", {format::type::bfyx, "sdpa_opt"}} }));
                }
            }
        } else {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                   {"sdpa", {format::type::bfyx, "sdpa_ref"}} }));
        }

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("input0", input0);
        net->set_input_data("input1", input1);
        net->set_input_data("input2", input2);
        net->set_input_data("input3", input3);

        auto outputs = net->execute();
        auto output = outputs.at("result").get_memory();
        return {output, net};
    }

    void execute(sdpa_test_params& p, bool is_caching_test = false) {
        const auto head_size = p.head_size;
        const auto num_heads = p.num_heads;
        const auto seq_length_q = p.sequence_length_q;
        const auto seq_length_kv = p.sequence_length_kv;
        const auto batch = p.batch;
        const auto use_scalar_scale_val = p.use_scalar_scale_val;
        const auto scale_val = p.scale_val;
        const auto use_scalar_attn_mask = p.use_scalar_attn_mask;
        const auto attn_mask_val = p.attn_mask_val;
        const auto test_two_rank_mask = p.sequence_length_q == p.sequence_length_kv ? true : false;

        auto& engine = get_test_engine();
        cldnn::layout input0_layout, input1_layout, input2_layout, input3_layout;
        cldnn::layout input0_static_layout, input1_static_layout, input2_static_layout, input3_static_layout;

        if (p.dynamic) {
            input0_layout = cldnn::layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
            input1_layout = cldnn::layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
            input2_layout = cldnn::layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);

            if (test_two_rank_mask) {
                input3_layout = cldnn::layout({ -1, -1}, data_types::f16, format::bfyx);
            } else {
                input3_layout = cldnn::layout({-1, num_heads, -1, -1}, data_types::f16, format::bfyx);
            }

            input0_static_layout = cldnn::layout({batch, seq_length_q,  num_heads, head_size}, data_types::f16, format::bfyx);
            input1_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
            input2_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
            if (test_two_rank_mask) {
                input3_static_layout = cldnn::layout({seq_length_q, seq_length_kv}, data_types::f32, format::bfyx);
            } else {
                input3_static_layout = cldnn::layout({batch, num_heads,     1,     seq_length_kv}, data_types::f16, format::bfyx);
            }
        } else {
            input0_static_layout = cldnn::layout({batch, seq_length_q,  num_heads, head_size}, data_types::f16, format::bfyx);
            input1_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
            input2_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);

            if (test_two_rank_mask) {
                input3_static_layout = cldnn::layout({seq_length_q, seq_length_kv}, data_types::f16, format::bfyx);
            } else {
                input3_static_layout = cldnn::layout({batch, num_heads,     1,     seq_length_kv}, data_types::f16, format::bfyx);
            }

            input0_layout = input0_static_layout;
            input1_layout = input1_static_layout;
            input2_layout = input2_static_layout;
            input3_layout = input3_static_layout;
        }

        auto input0 = engine.allocate_memory(input0_static_layout);
        auto input1 = engine.allocate_memory(input1_static_layout);
        auto input2 = engine.allocate_memory(input2_static_layout);
        auto input3 = engine.allocate_memory(input3_static_layout);

        load_input(input0, 0);
        load_input(input1, 1);
        load_input(input2, 2);
        load_input(input3, 3);

        auto [mem_ref_ptr, net_ref_ptr] = run_network(is_caching_test, false,
                                        input0_layout, input1_layout, input2_layout, input3_layout,
                                        input0, input1, input2, input3,
                                        use_scalar_scale_val, scale_val, use_scalar_attn_mask, attn_mask_val);
        auto [mem_opt_ptr, net_opt_ptr] = run_network(is_caching_test, true,
                                        input0_layout, input1_layout, input2_layout, input3_layout,
                                        input0, input1, input2, input3,
                                        use_scalar_scale_val, scale_val, use_scalar_attn_mask, attn_mask_val);

        if (is_caching_test) {
            auto inst = net_opt_ptr->get_primitive("sdpa");
            auto& sdpa_node = inst->get_node().as<scaled_dot_product_attention>();

            if (use_scalar_scale_val) {
                ASSERT_TRUE(sdpa_node.get_primitive()->scale_val.has_value());
                ASSERT_FLOAT_EQ(sdpa_node.get_primitive()->scale_val.value(), scale_val);
            }

            if (use_scalar_attn_mask) {
                ASSERT_TRUE(sdpa_node.get_primitive()->attn_mask_val.has_value());
                ASSERT_FLOAT_EQ(sdpa_node.get_primitive()->attn_mask_val.value(), attn_mask_val);
            }
        }

        cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_data(mem_ref_ptr, get_test_stream());
        cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_data(mem_opt_ptr, get_test_stream());
        {
            for (size_t idx = 0; idx < ref_data.size(); idx++) {
                ASSERT_FALSE(std::isnan(opt_data[idx]) || std::isnan(ref_data[idx])) << "NaN found at index " << idx;
            }
            auto ret = cosineSimilarity(ref_data, opt_data);
            ASSERT_GE(ret, 0.95f);
        }
    }

    static std::string
    PrintToStringParamName(const testing::TestParamInfo<sdpa_test_params>& info) {
        std::string result = "sdpa_gpu_test_" + std::to_string(info.param.head_size) + "_" +
               std::to_string(info.param.num_heads) + "_" +
               std::to_string(info.param.sequence_length_q) + "_" +
               std::to_string(info.param.sequence_length_kv) + "_" +
               std::to_string(info.param.batch);

        if (info.param.use_scalar_scale_val) {
            result += "_scale_" + std::to_string(static_cast<int>(info.param.scale_val * 1000));
        }

        if (info.param.use_scalar_attn_mask) {
            result += "_mask_" + std::to_string(static_cast<int>(info.param.attn_mask_val * 1000));
        }

        if (!info.param.dynamic) {
            result += "_static";
        }

        return result;
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_sdpa_gpu_test,
    sdpa_gpu_test,
    ::testing::Values(
        sdpa_test_params{64, 32, 990, 128, 2, true}, // dynamic
        sdpa_test_params{64, 32, 990, 128, 2, false}, // static
        sdpa_test_params{64, 32, 990, 1, 2, true}, // dynamic
        sdpa_test_params{64, 32, 990, 1, 2, false}, // static
        sdpa_test_params{64, 10, 77, 77, 1, true}, // two ranks mask
        sdpa_test_params{64, 10, 77, 77, 1, false}, // two ranks mask
        sdpa_test_params{64, 32, 128, 128, 2, true, 0.125f, false, 0.0f},  // scale_val only
        sdpa_test_params{64, 32, 128, 128, 2, false, 1.0f, true, 0.5f},     // attn_mask only
        sdpa_test_params{512, 8, 1, 1024, 2, true}
    ),
    sdpa_gpu_test::PrintToStringParamName
);

TEST_P(sdpa_gpu_test, basic) {
    auto p = GetParam();
    execute(p);
}

TEST_P(sdpa_gpu_test, basic_caching) {
    auto p = GetParam();
    execute(p, true);
}

// ---------------------------------------------------------------------------
// Compressed (int8 / int4) KV-cache SDPA tests.
//
// The optimized SDPA kernel (sdpa_opt) dequantizes the KV cache on read using
// the asymmetric formula `deq = (q - zp) * scale`, where a single (scale, zp)
// pair is shared by all head_size channels of a given (batch, head, token) and
// stored interleaved as [scale, zp] (InterleavedScalesZP) in fp16. INT4 packs
// two unsigned nibbles per byte: low nibble -> even head dim, high nibble -> odd
// head dim.
//
// These tests build a `scaled_dot_product_attention` primitive with
// is_kv_compressed=true directly and feed host-quantized KV plus the matching
// scale/zp buffers to the forced `sdpa_opt` path. The golden reference is an
// independent float attention (uncompressed, forced `sdpa_ref`) fed the exact
// host-dequantized KV, so the two networks operate on identical effective KV
// and any divergence isolates the kernel's dequant + attention math.
// ---------------------------------------------------------------------------
struct kv_quant_result {
    std::vector<int8_t> packed;            // quantized (INT4: two nibbles per byte)
    std::vector<ov::float16> dequantized;  // (q - zp) * scale, full head_size
    std::vector<ov::float16> scales;       // per-(batch, head, channel) scale, [batch, heads, head_size]
    std::vector<ov::float16> zero_points;  // per-(batch, head, channel) zp, same layout (asymmetric only)
};

// Asymmetric per-(batch, head, token) quantization over the whole head_size dim.
// Layout of `src` is bfyx {batch, seq, heads, head_size} (matches transpose {0,2,1,3}).
// Load a raw little-endian binary dump into a typed vector (no header parsing).
template <typename T>
static std::vector<T> load_bin_as(const std::string& path) {
    const auto bytes = ov::util::load_binary(path);
    OPENVINO_ASSERT(!bytes.empty(), "Failed to load (missing/empty) binary file: ", path);
    OPENVINO_ASSERT(bytes.size() % sizeof(T) == 0, "Binary file size is not a multiple of element size: ", path);
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

// Per-channel KV quantization: one (scale, zp) per (batch, head, channel), shared across the
// whole sequence dimension. When seq_major is true the source is laid out
// [batch, seq, heads, head_size]; when false it is [batch, heads, seq, head_size]. The produced
// scales / zero_points buffers are always laid out [batch, heads, head_size].
//
// Two quantization strategies are supported:
//   * Symmetric  (symmetric=true):  signed codes centered on zero, deq = q * scale (no
//     zero-point). The per-channel scale is max_abs / q_max over all sequence positions.
//   * Asymmetric (symmetric=false): unsigned codes with a zero-point, deq = (q - zp) * scale.
//     The per-channel scale/zp are derived from the [min, max] range over all sequence
//     positions: scale = (max - min) / (q_max - q_min), zp = q_min - min / scale.
// In both cases the derived scale is multiplied by a per-channel random factor (see below) so the
// scales vary between neighbouring channels rather than exactly fitting the data range.
// provided_scale, when non-null, supplies the per-channel scales ([batch, heads, head_size])
// instead of deriving (and randomizing) them; the zero-point is still derived for the asymmetric
// case so it matches the given scale.
static kv_quant_result quantize_kv_per_channel(const std::vector<ov::float16>& src,
                                               int batch,
                                               int seq,
                                               int heads,
                                               int head_size,
                                               int bit_width,
                                               bool seq_major = true,
                                               const std::vector<ov::float16>* provided_scale = nullptr,
                                               bool symmetric = false) {
    // Symmetric uses a signed range centered on zero (no zero-point); asymmetric int4 uses
    // unsigned nibbles [0, 15] with a zero-point.
    const int q_min = symmetric ? -(bit_width == 4 ? 7 : 127) : (bit_width == 4 ? 0 : -128);
    const int q_max = symmetric ? (bit_width == 4 ? 7 : 127) : (bit_width == 4 ? 15 : 127);
    const int packed_hs = bit_width == 4 ? head_size / 2 : head_size;
    const size_t channel_count = static_cast<size_t>(batch) * heads * head_size;

    kv_quant_result r;
    r.packed.assign(static_cast<size_t>(batch) * seq * heads * packed_hs, 0);
    r.dequantized.assign(static_cast<size_t>(batch) * seq * heads * head_size, ov::float16(0.0f));
    r.scales.assign(channel_count, ov::float16(0.0f));
    if (!symmetric)
        r.zero_points.assign(channel_count, ov::float16(0.0f));

    const auto elem_base = [&](int b, int s, int h) -> size_t {
        return seq_major ? ((static_cast<size_t>(b) * seq + s) * heads + h) * head_size : ((static_cast<size_t>(b) * heads + h) * seq + s) * head_size;
    };
    const auto packed_base_of = [&](int b, int s, int h) -> size_t {
        return seq_major ? ((static_cast<size_t>(b) * seq + s) * heads + h) * packed_hs : ((static_cast<size_t>(b) * heads + h) * seq + s) * packed_hs;
    };
    const auto pack_value = [&](size_t packed_base, int d, int q) {
        if (bit_width == 4) {
            // low nibble = even head dim, high nibble = odd head dim
            auto& byte = r.packed[packed_base + d / 2];
            if (d % 2 == 0)
                byte = static_cast<int8_t>((byte & 0xF0) | (q & 0x0F));
            else
                byte = static_cast<int8_t>((byte & 0x0F) | ((q & 0x0F) << 4));
        } else {
            r.packed[packed_base + d] = static_cast<int8_t>(q);
        }
    };

    // Random per-channel multiplier applied on top of the derived scale, generated the same way as
    // the tests' random inputs (1/8-resolution values in [1, 2]). Neighbouring channels therefore
    // get clearly different scales, so a kernel that mis-indexes the scale buffer diverges from the
    // host dequantization instead of silently agreeing; staying >= 1 keeps every value inside the
    // quantized range. The fixed seed keeps the test reproducible.
    tests::random_generator scale_rg("quantize_kv_per_channel_scale");
    const auto scale_mul = scale_rg.generate_random_1d<float>(channel_count, 1, 2, 8);

    // Derive (or take) one scale and zero-point per (batch, head, channel) from the values of
    // that channel across the whole sequence.
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < head_size; ++d) {
                const size_t ch = (static_cast<size_t>(b) * heads + h) * head_size + d;
                float min_v = std::numeric_limits<float>::max();
                float max_v = std::numeric_limits<float>::lowest();
                for (int s = 0; s < seq; ++s) {
                    const float v = static_cast<float>(src[elem_base(b, s, h) + d]);
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                }
                float scale = 0.0f;
                if (provided_scale != nullptr) {
                    scale = static_cast<float>((*provided_scale)[ch]);
                } else if (symmetric) {
                    scale = std::max(std::abs(min_v), std::abs(max_v)) / static_cast<float>(q_max) * scale_mul[ch];
                } else {
                    scale = (max_v - min_v) / static_cast<float>(q_max - q_min) * scale_mul[ch];
                }
                if (scale == 0.0f)
                    scale = 1.0f;  // degenerate (constant) channel: avoid div-by-zero
                r.scales[ch] = ov::float16(scale);
                if (!symmetric) {
                    // deq = (q - zp) * scale, so min_v maps to q_min: zp = q_min - min_v / scale.
                    r.zero_points[ch] = ov::float16(std::round(static_cast<float>(q_min) - min_v / scale));
                }
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            for (int h = 0; h < heads; ++h) {
                const size_t base = elem_base(b, s, h);
                const size_t packed_base = packed_base_of(b, s, h);
                const size_t scale_base = (static_cast<size_t>(b) * heads + h) * head_size;
                for (int d = 0; d < head_size; ++d) {
                    // Quantize through the fp16-rounded scale/zp so the host dequantization
                    // matches exactly what the kernel reads from the scale/zp buffers.
                    const float scale = static_cast<float>(r.scales[scale_base + d]);
                    const float zp = symmetric ? 0.0f : static_cast<float>(r.zero_points[scale_base + d]);
                    const float v = static_cast<float>(src[base + d]);
                    int q = static_cast<int>(std::lround(v / scale + zp));
                    q = std::max(q_min, std::min(q_max, q));
                    r.dequantized[base + d] = ov::float16((static_cast<float>(q) - zp) * scale);
                    pack_value(packed_base, d, q);
                }
            }
        }
    }
    return r;
}

static void run_compressed_kv_sdpa_test(const sdpa_test_params& params) {
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int bit_width = params.bit_width;
    const bool asymmetric = params.asymmetric;
    const int batch = params.batch;
    const int q_num_heads = params.num_heads;
    const int kv_num_heads = params.kv_num_heads;
    const int seq_q = params.sequence_length_q;
    const int seq_kv = params.sequence_length_kv;
    const int head_size = params.head_size;

    OPENVINO_ASSERT(bit_width != 4 || head_size % 2 == 0, "INT4 KV packing requires an even head size");
    const int packed_head_size = bit_width == 4 ? head_size / 2 : head_size;

    // Q and original float K/V use [batch, heads, sequence, head_size].
    auto q_data = rg.generate_random_1d<ov::float16>(static_cast<size_t>(batch) * seq_q * q_num_heads * head_size, -1.0f, 1.0f);
    auto k_orig = rg.generate_random_1d<ov::float16>(static_cast<size_t>(batch) * seq_kv * kv_num_heads * head_size, -1.0f, 1.0f);
    auto v_orig = rg.generate_random_1d<ov::float16>(static_cast<size_t>(batch) * seq_kv * kv_num_heads * head_size, -1.0f, 1.0f);

    auto k_q = quantize_kv_per_channel(k_orig,
                                       batch,
                                       seq_kv,
                                       kv_num_heads,
                                       head_size,
                                       bit_width,
                                       /*seq_major=*/false,
                                       nullptr,
                                       /*symmetric=*/!asymmetric);
    auto v_q = quantize_kv_per_channel(v_orig,
                                       batch,
                                       seq_kv,
                                       kv_num_heads,
                                       head_size,
                                       bit_width,
                                       /*seq_major=*/false,
                                       nullptr,
                                       /*symmetric=*/!asymmetric);

    const layout q_layout({batch, q_num_heads, seq_q, head_size}, data_types::f16, format::bfyx);
    const layout kv_deq_layout({batch, kv_num_heads, seq_kv, head_size}, data_types::f16, format::bfyx);
    // INT4 stores two adjacent head-dimension values in each byte: [B, H, S, D/2].
    const layout kv_packed_layout({batch, kv_num_heads, seq_kv, packed_head_size}, data_types::i8, format::bfyx);
    const layout comp_layout({batch, kv_num_heads, 1, head_size}, data_types::f16, format::bfyx);

    const ov::Shape expected_packed_shape = {static_cast<size_t>(batch),
                                             static_cast<size_t>(kv_num_heads),
                                             static_cast<size_t>(seq_kv),
                                             static_cast<size_t>(packed_head_size)};
    ASSERT_EQ(kv_packed_layout.get_shape(), expected_packed_shape);
    ASSERT_EQ(k_q.packed.size(), ov::shape_size(expected_packed_shape));
    ASSERT_EQ(v_q.packed.size(), ov::shape_size(expected_packed_shape));

    auto q_mem = engine.allocate_memory(q_layout);
    set_values(q_mem, q_data);

    // --- Golden reference: uncompressed float attention on host-dequantized KV (sdpa_ref) ---
    auto make_ref_output = [&]() {
        auto k_mem = engine.allocate_memory(kv_deq_layout);
        auto v_mem = engine.allocate_memory(kv_deq_layout);
        set_values(k_mem, k_q.dequantized);
        set_values(v_mem, v_q.dequantized);

        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", kv_deq_layout));
        topo.add(input_layout("v", kv_deq_layout));
        auto prim = scaled_dot_product_attention("sdpa",
                                                 {input_info("q"), input_info("k"), input_info("v")},
                                                 true,
                                                 -1,
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {},
                                                 false);
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        return net->execute().at("result").get_memory();
    };

    // --- Compressed path: quantized KV + interleaved scale/zp on the optimized kernel (sdpa_opt) ---
    auto make_opt_output = [&]() {
        auto k_mem = engine.allocate_memory(kv_packed_layout);
        auto v_mem = engine.allocate_memory(kv_packed_layout);
        auto k_comp_mem = engine.allocate_memory(comp_layout);
        auto v_comp_mem = engine.allocate_memory(comp_layout);
        auto k_zp_mem = asymmetric ? engine.allocate_memory(comp_layout) : nullptr;
        auto v_zp_mem = asymmetric ? engine.allocate_memory(comp_layout) : nullptr;
        set_values(k_mem, k_q.packed);
        set_values(v_mem, v_q.packed);
        set_values(k_comp_mem, k_q.scales);
        set_values(v_comp_mem, v_q.scales);
        if (asymmetric) {
            set_values(k_zp_mem, k_q.zero_points);
            set_values(v_zp_mem, v_q.zero_points);
        }

        scaled_dot_product_attention::QuantizationAttributes qa;
        qa.quantization_type =
            asymmetric ? ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric : ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
        qa.quantization_dt = bit_width == 4 ? (asymmetric ? ov::element::u4 : ov::element::i4) : ov::element::i8;
        qa.scale_dt = ov::element::f16;
        qa.zp_dt = ov::element::f16;
        qa.group_sizes = {1, 1, UINT64_MAX, 1};
        qa.scales_zp_output_order = {0, 1, 2, 3};
        qa.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", kv_packed_layout));
        topo.add(input_layout("v", kv_packed_layout));
        topo.add(input_layout("k_scale", comp_layout));
        topo.add(input_layout("v_scale", comp_layout));
        std::vector<input_info> inputs = {input_info("q"), input_info("k"), input_info("v"), input_info("k_scale"), input_info("v_scale")};
        if (asymmetric) {
            topo.add(input_layout("k_zp", comp_layout));
            topo.add(input_layout("v_zp", comp_layout));
            inputs.emplace_back("k_zp");
            inputs.emplace_back("v_zp");
        }

        auto prim = scaled_dot_product_attention("sdpa", inputs, true, -1, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, qa, true);
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::hint::kv_cache_precision(bit_width == 4 ? ov::element::i4 : ov::element::i8));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"sdpa", {format::type::bfyx, "sdpa_opt"}}}));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        net->set_input_data("k_scale", k_comp_mem);
        net->set_input_data("v_scale", v_comp_mem);
        if (asymmetric) {
            net->set_input_data("k_zp", k_zp_mem);
            net->set_input_data("v_zp", v_zp_mem);
        }
        return net->execute().at("result").get_memory();
    };

    auto ref_mem = make_ref_output();
    auto opt_mem = make_opt_output();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_ptr(ref_mem, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_ptr(opt_mem, get_test_stream());

    ASSERT_EQ(ref_ptr.size(), opt_ptr.size());
    for (size_t i = 0; i < opt_ptr.size(); ++i) {
        ASSERT_FALSE(std::isnan(static_cast<float>(opt_ptr[i]))) << "NaN in compressed output at index " << i;
    }
    const float sim = cosineSimilarity(ref_ptr, opt_ptr);
    // Opt-in debug dumps: set env var `dumpdata` to any value other than "false" to enable.
    static const bool dumpdata = []() {
        const char* txt = std::getenv("dumpdata");
        return txt != nullptr && std::strcmp(txt, "false") != 0;
    }();
    if (dumpdata) {
        std::cout << "[fake data] result size=" << opt_ptr.size() << " cosine_sim=" << sim << " cosine_sim_mask=" << sim << std::endl;
        std::cout << "[fake data] ref[0..7]:";
        for (size_t i = 0; i < std::min<size_t>(8, ref_ptr.size()); ++i)
            std::cout << " " << static_cast<float>(ref_ptr[i]);
        std::cout << "\n[fake data] opt[0..7]:";
        for (size_t i = 0; i < std::min<size_t>(8, opt_ptr.size()); ++i)
            std::cout << " " << static_cast<float>(opt_ptr[i]);
        std::cout << std::endl;

        // Dump both results to txt: first line is the cldnn tensor shape, then one value per line.
        auto dump_result_txt = [](const std::string& fname, const memory::ptr& mem, cldnn::mem_lock<ov::float16, mem_lock_type::read>& lock) {
            std::ofstream fs(fname);
            fs << "shape: " << mem->get_layout().get_tensor().to_string() << " " << std::endl;
            for (size_t i = 0; i < lock.size(); ++i)
                fs << static_cast<float>(lock[i]) << std::endl;
            std::cout << "[fake data] wrote " << lock.size() << " values to " << fname << std::endl;
        };
        dump_result_txt("fake data_ref_" + std::to_string(bit_width) + ".txt", ref_mem, ref_ptr);
        dump_result_txt("fake data_opt_" + std::to_string(bit_width) + ".txt", opt_mem, opt_ptr);
    }
    ASSERT_GE(sim, 0.95f) << "bit_width=" << bit_width << ", asymmetric=" << asymmetric << " cosine similarity too low: " << sim;
}

static void run_compressed_kv_sdpa_gqa_test(int bit_width) {
    auto& engine = get_test_engine();

    const int batch = 1;
    const int q_num_heads = 40;
    const int kv_num_heads = 10;
    const int seq_q = 512;
    const int seq_kv = 512;
    const int head_size = 128;
    const int packed_hs = bit_width == 4 ? head_size / 2 : head_size;

    // Q and original (float) K/V loaded from external dumps. On-disk layout matches the file
    // names: [batch, heads, seq, head_size] (bfyx with heads in the feature dim). Edit data_dir
    // to point at the directory holding the .bin dumps.
    const std::string data_dir = "test_data/";  // TODO: set to your data directory
    auto q_data = load_bin_as<ov::float16>(data_dir + "q_f16__1_40_512_128__bfyx.bin");
    auto k_orig = load_bin_as<ov::float16>(data_dir + "k_f16__1_10_512_128__bfyx.bin");
    auto v_orig = load_bin_as<ov::float16>(data_dir + "v_f16__1_10_512_128__bfyx.bin");

    // Externally provided per-(batch, head, channel) symmetric quantization scales
    // ([batch, heads, head_size]); the sequence dimension shares the same scale. The dumps are
    // stored as fp32 on disk but consumed as fp16 scales, so load fp32 and round to fp16.
    const auto load_scale_fp16 = [](const std::string& path) {
        const auto raw = load_bin_as<float>(path);
        std::vector<ov::float16> out(raw.size());
        for (size_t i = 0; i < raw.size(); ++i)
            out[i] = ov::float16(raw[i]);
        return out;
    };
    auto k_scale = load_scale_fp16(data_dir + "k_scale_0.bin");
    auto v_scale = load_scale_fp16(data_dir + "v_scale_0.bin");

    // Opt-in debug dumps: set env var `dumpdata` to any value other than "false" to enable.
    static const bool dumpdata = []() {
        const char* txt = std::getenv("dumpdata");
        return txt != nullptr && std::strcmp(txt, "false") != 0;
    }();

    if (dumpdata) {
        std::cout << "[gqa] sizes q_data=" << q_data.size() << " k_orig=" << k_orig.size() << " v_orig=" << v_orig.size() << " k_scale=" << k_scale.size()
                  << " v_scale=" << v_scale.size() << std::endl;
        std::cout << "[gqa] expected: qkv_per_head=" << (static_cast<size_t>(batch) * kv_num_heads * seq_kv * head_size)
                  << " scale_per_token=" << (static_cast<size_t>(batch) * kv_num_heads * seq_kv)
                  << " scale_per_channel=" << (static_cast<size_t>(batch) * kv_num_heads * head_size) << std::endl;
        std::cout << "[gqa] k_orig[0..7]:";
        for (size_t i = 0; i < std::min<size_t>(8, k_orig.size()); ++i)
            std::cout << " " << static_cast<float>(k_orig[i]);
        std::cout << "\n[gqa] k_scale[0..7]:";
        for (size_t i = 0; i < std::min<size_t>(8, k_scale.size()); ++i)
            std::cout << " " << static_cast<float>(k_scale[i]);
        std::cout << std::endl;
    }

    ASSERT_EQ(q_data.size(), static_cast<size_t>(batch) * q_num_heads * seq_q * head_size);
    ASSERT_EQ(k_orig.size(), static_cast<size_t>(batch) * kv_num_heads * seq_kv * head_size);
    ASSERT_EQ(v_orig.size(), static_cast<size_t>(batch) * kv_num_heads * seq_kv * head_size);
    ASSERT_EQ(k_scale.size(), static_cast<size_t>(batch) * kv_num_heads * head_size);
    ASSERT_EQ(v_scale.size(), static_cast<size_t>(batch) * kv_num_heads * head_size);

    // Compute packed KV and dequantized values on host from the unpacked K/V using the externally
    // provided per-channel symmetric scales (no zero-point), one scale per (batch, head, channel)
    // shared across the sequence. seq_major=false because the dumps are laid out
    // [batch, heads, seq, head_size].
    auto k_q = quantize_kv_per_channel(k_orig, batch, seq_kv, kv_num_heads, head_size, bit_width, /*seq_major=*/false, &k_scale, /*symmetric=*/true);
    auto v_q = quantize_kv_per_channel(v_orig, batch, seq_kv, kv_num_heads, head_size, bit_width, /*seq_major=*/false, &v_scale, /*symmetric=*/true);

    if (dumpdata) {
        std::cout << "[gqa] k_packed=" << k_q.packed.size() << " k_dequant=" << k_q.dequantized.size() << std::endl;
        std::cout << "[gqa] k_packed[0..7]:";
        for (size_t i = 0; i < std::min<size_t>(8, k_q.packed.size()); ++i)
            std::cout << " " << static_cast<int>(k_q.packed[i]);
        std::cout << "\n[gqa] k_dequant[0..7]:";
        for (size_t i = 0; i < std::min<size_t>(8, k_q.dequantized.size()); ++i)
            std::cout << " " << static_cast<float>(k_q.dequantized[i]);
        std::cout << std::endl;

        // Dump the host-quantized K/V to txt: packed integer codes and dequantized values,
        // one value per line, written to <prefix>_packed.txt and <prefix>_dequant.txt.
        auto dump_quant_txt = [](const std::string& prefix, const kv_quant_result& q) {
            std::ofstream packed_fs(prefix + "_packed.txt");
            for (size_t i = 0; i < q.packed.size(); ++i)
                packed_fs << static_cast<int>(q.packed[i]) << std::endl;
            std::ofstream dequant_fs(prefix + "_dequant.txt");
            for (size_t i = 0; i < q.dequantized.size(); ++i)
                dequant_fs << static_cast<float>(q.dequantized[i]) << std::endl;
            std::cout << "[gqa] wrote " << q.packed.size() << " packed / " << q.dequantized.size() << " dequantized values to " << prefix
                      << "_{packed,dequant}.txt" << std::endl;
        };
        dump_quant_txt("k_q_" + std::to_string(bit_width), k_q);
        dump_quant_txt("v_q_" + std::to_string(bit_width), v_q);
        std::cout << std::endl;
    }

    // Physical layouts follow the on-disk order [batch, heads, seq, head_size], so the SDPA
    // transpose orders are identity {0, 1, 2, 3}.
    const layout q_layout({batch, q_num_heads, seq_q, head_size}, data_types::f16, format::bfyx);
    const layout kv_deq_layout({batch, kv_num_heads, seq_kv, head_size}, data_types::f16, format::bfyx);
    const layout kv_packed_layout({batch, kv_num_heads, seq_kv, packed_hs}, data_types::i8, format::bfyx);
    // Per-channel scales: one value per (batch, head, channel), shared across the sequence.
    const layout comp_layout({batch, kv_num_heads, 1, head_size}, data_types::f16, format::bfyx);

    auto q_mem = engine.allocate_memory(q_layout);
    set_values(q_mem, q_data);

    // --- Golden reference: uncompressed float attention on host-dequantized KV (sdpa_ref) ---
    auto make_ref_output = [&]() {
        auto k_mem = engine.allocate_memory(kv_deq_layout);
        auto v_mem = engine.allocate_memory(kv_deq_layout);
        set_values(k_mem, k_q.dequantized);
        set_values(v_mem, v_q.dequantized);

        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", kv_deq_layout));
        topo.add(input_layout("v", kv_deq_layout));
        auto prim = scaled_dot_product_attention("sdpa",
                                                 {input_info("q"), input_info("k"), input_info("v")},
                                                 true,
                                                 -1,
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {},
                                                 false);
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        return net->execute().at("result").get_memory();
    };

    // --- Compressed path: quantized KV + interleaved scale/zp on the optimized kernel (sdpa_opt) ---
    auto make_opt_output = [&]() {
        auto k_mem = engine.allocate_memory(kv_packed_layout);
        auto v_mem = engine.allocate_memory(kv_packed_layout);
        auto k_comp_mem = engine.allocate_memory(comp_layout);
        auto v_comp_mem = engine.allocate_memory(comp_layout);
        set_values(k_mem, k_q.packed);
        set_values(v_mem, v_q.packed);
        set_values(k_comp_mem, k_q.scales);
        set_values(v_comp_mem, v_q.scales);

        scaled_dot_product_attention::QuantizationAttributes qa;
        qa.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
        qa.quantization_dt = ov::element::i8;
        qa.scale_dt = ov::element::f16;
        qa.group_sizes = {1, 1, UINT64_MAX, 1};
        qa.scales_zp_output_order = {0, 1, 2, 3};
        qa.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", kv_packed_layout));
        topo.add(input_layout("v", kv_packed_layout));
        topo.add(input_layout("k_scale", comp_layout));
        topo.add(input_layout("v_scale", comp_layout));

        // Inputs: Q, K, V, key_scales, value_scales (symmetric Planar storage, no zero-point).
        auto prim = scaled_dot_product_attention("sdpa",
                                                 {input_info("q"), input_info("k"), input_info("v"), input_info("k_scale"), input_info("v_scale")},
                                                 true,
                                                 -1,
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 {0, 1, 2, 3},
                                                 qa,
                                                 true);
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::hint::kv_cache_precision(bit_width == 4 ? ov::element::i4 : ov::element::i8));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"sdpa", {format::type::bfyx, "sdpa_opt"}}}));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        net->set_input_data("k_scale", k_comp_mem);
        net->set_input_data("v_scale", v_comp_mem);
        return net->execute().at("result").get_memory();
    };

    auto ref_mem = make_ref_output();
    auto opt_mem = make_opt_output();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_ptr(ref_mem, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_ptr(opt_mem, get_test_stream());

    ASSERT_EQ(ref_ptr.size(), opt_ptr.size());
    for (size_t i = 0; i < opt_ptr.size(); ++i) {
        ASSERT_FALSE(std::isnan(static_cast<float>(opt_ptr[i]))) << "NaN in compressed output at index " << i;
    }
    const float sim = cosineSimilarity(ref_ptr, opt_ptr);

    if (dumpdata) {
        std::cout << "[gqa] result size=" << opt_ptr.size() << " cosine_sim=" << sim << std::endl;
        std::cout << "[gqa] ref[0..7]:";
        for (size_t i = 0; i < std::min<size_t>(8, ref_ptr.size()); ++i)
            std::cout << " " << static_cast<float>(ref_ptr[i]);
        std::cout << "\n[gqa] opt[0..7]:";
        for (size_t i = 0; i < std::min<size_t>(8, opt_ptr.size()); ++i)
            std::cout << " " << static_cast<float>(opt_ptr[i]);
        std::cout << std::endl;

        // Dump both results to txt: first line is the cldnn tensor shape, then one value per line.
        auto dump_result_txt = [](const std::string& fname, const memory::ptr& mem, cldnn::mem_lock<ov::float16, mem_lock_type::read>& lock) {
            std::ofstream fs(fname);
            fs << "shape: " << mem->get_layout().get_tensor().to_string() << " " << std::endl;
            for (size_t i = 0; i < lock.size(); ++i)
                fs << static_cast<float>(lock[i]) << std::endl;
            std::cout << "[gqa] wrote " << lock.size() << " values to " << fname << std::endl;
        };
        dump_result_txt("gqa_ref_" + std::to_string(bit_width) + ".txt", ref_mem, ref_ptr);
        dump_result_txt("gqa_opt_" + std::to_string(bit_width) + ".txt", opt_mem, opt_ptr);
    }

    ASSERT_GE(sim, 0.95f) << "bit_width=" << bit_width << " cosine similarity too low: " << sim;
}

struct sdpa_gpu_compressed_kv_test : public ::testing::TestWithParam<sdpa_test_params> {
    static std::string PrintToStringParamName(const testing::TestParamInfo<sdpa_test_params>& info) {
        const auto& p = info.param;
        return std::string("int") + std::to_string(p.bit_width) + (p.asymmetric ? "_asymmetric_" : "_symmetric_") +
               (p.num_heads == p.kv_num_heads ? "mha_" : "gqa_") + (p.sequence_length_q == 1 ? "decode" : "prefill");
    }
};

TEST_P(sdpa_gpu_compressed_kv_test, compare_with_gpu_dequantized_reference) {
    run_compressed_kv_sdpa_test(GetParam());
}

INSTANTIATE_TEST_SUITE_P(smoke_sdpa_gpu_compressed_kv,
                         sdpa_gpu_compressed_kv_test,
                         ::testing::Values(sdpa_test_params{128, 40, 10, 512, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 4, false},
                                           sdpa_test_params{128, 40, 10, 512, 512, 1, 4, true},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 4, true},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 4, true},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 4, true},
                                           sdpa_test_params{128, 40, 10, 512, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 8, false},
                                           sdpa_test_params{128, 40, 10, 512, 512, 1, 8, true},
                                           sdpa_test_params{128, 40, 10, 1, 512, 1, 8, true},
                                           sdpa_test_params{128, 40, 40, 512, 512, 1, 8, true},
                                           sdpa_test_params{128, 40, 40, 1, 512, 1, 8, true}),
                         sdpa_gpu_compressed_kv_test::PrintToStringParamName);

TEST(sdpa_gpu_compressed_kv_gqa, int8) {
    run_compressed_kv_sdpa_gqa_test(8);
}
#endif

TEST(sdpa_gpu_custom, dynamic_mismatched_v_head_size) {
    auto& engine = get_test_engine();

    const ov::PartialShape qk_shape{-1, 1, -1, 384};
    const ov::PartialShape v_shape{-1, 1, -1, -1};
    const ov::Shape qk_static_shape{1, 1, 16, 384};
    const ov::Shape v_static_shape{1, 1, 16, 256};

    const layout q_layout(qk_shape, data_types::f16, format::bfyx);
    const layout k_layout(qk_shape, data_types::f16, format::bfyx);
    const layout v_layout(v_shape, data_types::f16, format::bfyx);
    const layout qk_static_layout(qk_static_shape, data_types::f16, format::bfyx);
    const layout v_static_layout(v_static_shape, data_types::f16, format::bfyx);

    auto q_mem = engine.allocate_memory(qk_static_layout);
    auto k_mem = engine.allocate_memory(qk_static_layout);
    auto v_mem = engine.allocate_memory(v_static_layout);

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto fill_random = [&](const memory::ptr& mem) {
        auto data = rg.generate_random_1d<ov::float16>(mem->get_layout().count(), -1.0f, 1.0f);
        set_values(mem, data);
    };
    fill_random(q_mem);
    fill_random(k_mem);
    fill_random(v_mem);

    topology topology;
    topology.add(input_layout("q", q_layout));
    topology.add(input_layout("k", k_layout));
    topology.add(input_layout("v", v_layout));
    topology.add(scaled_dot_product_attention("sdpa",
                                              {input_info("q"), input_info("k"), input_info("v")},
                                              false,
                                              -1,
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {},
                                              false));
    topology.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

    auto run_network = [&](bool force_ref) {
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        if (force_ref) {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                {"sdpa", {format::type::bfyx, "sdpa_ref"}}
            }));
        }

        auto network = get_network(engine, topology, config, get_test_stream_ptr(), false);
        network->set_input_data("q", q_mem);
        network->set_input_data("k", k_mem);
        network->set_input_data("v", v_mem);
        auto output = network->execute().at("result").get_memory();
        return std::make_pair(network, output);
    };

    auto [network, output] = run_network(false);
    auto [ref_network, ref_output] = run_network(true);

    ASSERT_NE(network->get_primitive_info("sdpa").find("sdpa_ref"), std::string::npos);
    ASSERT_EQ(output->get_layout().get_shape(), v_static_shape);

    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_data(output, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_output_data(ref_output, get_test_stream());
    ASSERT_EQ(output_data.size(), ref_output_data.size());
    for (size_t i = 0; i < output_data.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(output_data[i]), static_cast<float>(ref_output_data[i]), 1e-3f)
            << "Mismatch at index " << i;
    }
}

TEST(sdpa_gpu_custom, different_rank_orders_dynamic_v_head_size_rejects_opt) {
    auto& engine = get_test_engine();

    auto q_prim = std::make_shared<input_layout>("q", layout(ov::PartialShape{-1, -1, 384}, data_types::f16, format::bfyx));
    auto k_prim = std::make_shared<input_layout>("k", layout(ov::PartialShape{-1, -1, 384}, data_types::f16, format::bfyx));
    auto v_prim = std::make_shared<input_layout>("v", layout(ov::PartialShape{-1, 1, 16, -1}, data_types::f16, format::bfyx));
    auto sdpa_prim = std::make_shared<scaled_dot_product_attention>("sdpa",
                                                                   std::vector{input_info("q"), input_info("k"), input_info("v")},
                                                                   false,
                                                                   -1,
                                                                   std::vector<int64_t>{0, 1, 2},
                                                                   std::vector<int64_t>{0, 1, 2},
                                                                   std::vector<int64_t>{0, 1, 2, 3},
                                                                   std::vector<int64_t>{0, 1, 2, 3},
                                                                   scaled_dot_product_attention::QuantizationAttributes{},
                                                                   false);

    program prog(engine);
    auto& q_node = prog.get_or_create(q_prim);
    auto& k_node = prog.get_or_create(k_prim);
    auto& v_node = prog.get_or_create(v_prim);
    auto& sdpa_node = prog.get_or_create(sdpa_prim);
    program_wrapper::add_connection(prog, q_node, sdpa_node);
    program_wrapper::add_connection(prog, k_node, sdpa_node);
    program_wrapper::add_connection(prog, v_node, sdpa_node);
    sdpa_node.recalc_output_layout();

    ov::intel_gpu::ocl::SDPAOpt sdpa_opt(shape_types::dynamic_shape);
    EXPECT_FALSE(sdpa_opt.validate_impl(sdpa_node));
}

TEST(sdpa_gpu_custom, static_zero_dimension_throws) {
    auto& engine = get_test_engine();

    const std::vector<std::array<ov::Shape, 3>> input_shapes = {
        {{{1, 0, 16, 32}, {1, 0, 16, 32}, {1, 0, 16, 32}}},
        {{{1, 1, 16, 32}, {1, 1, 16, 32}, {1, 1, 16, 0}}},
    };

    for (const auto& shapes : input_shapes) {
        topology topology;
        topology.add(input_layout("q", layout(shapes[0], data_types::f16, format::bfyx)));
        topology.add(input_layout("k", layout(shapes[1], data_types::f16, format::bfyx)));
        topology.add(input_layout("v", layout(shapes[2], data_types::f16, format::bfyx)));
        topology.add(scaled_dot_product_attention("sdpa",
                                                  {input_info("q"), input_info("k"), input_info("v")},
                                                  false,
                                                  -1,
                                                  {0, 1, 2, 3},
                                                  {0, 1, 2, 3},
                                                  {0, 1, 2, 3},
                                                  {0, 1, 2, 3},
                                                  {},
                                                  false));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
            {"sdpa", {format::type::bfyx, "sdpa_ref"}}
        }));

        EXPECT_ANY_THROW(get_network(engine, topology, config, get_test_stream_ptr(), false));
    }
}

TEST(sdpa_gpu_custom, dynamic_zero_head_size_throws) {
    auto& engine = get_test_engine();

    const ov::PartialShape dynamic_shape{-1, 1, -1, -1};
    const layout dynamic_layout(dynamic_shape, data_types::f16, format::bfyx);
    auto q_mem = engine.allocate_memory(layout({1, 1, 16, 0}, data_types::f16, format::bfyx));
    auto k_mem = engine.allocate_memory(layout({1, 1, 16, 0}, data_types::f16, format::bfyx));
    auto v_mem = engine.allocate_memory(layout({1, 1, 16, 32}, data_types::f16, format::bfyx));

    topology topology;
    topology.add(input_layout("q", dynamic_layout));
    topology.add(input_layout("k", dynamic_layout));
    topology.add(input_layout("v", dynamic_layout));
    topology.add(scaled_dot_product_attention("sdpa",
                                              {input_info("q"), input_info("k"), input_info("v")},
                                              false,
                                              -1,
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {0, 1, 2, 3},
                                              {},
                                              false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"sdpa", {format::type::bfyx, "sdpa_ref"}}
    }));

    auto network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    network->set_input_data("q", q_mem);
    network->set_input_data("k", k_mem);
    network->set_input_data("v", v_mem);
    try {
        network->execute();
        FAIL() << "Expected runtime SDPA dimension validation to throw";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("invalid non-positive q_head_size for runtime dispatch"), std::string::npos)
            << "Unexpected exception message: " << e.what();
    }
}

TEST(sdpa_gpu_custom, single_token_cond_attn_mask_clamp) {
    tests::random_generator rg; rg.set_seed(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (engine.get_device_info().supports_immad) {
        return;
    }

    const int head_size = 32;
    const int num_heads = 1;
    const int seq_length_q = 1;
    const int seq_length_kv = 448;
    const int batch = 1;

    layout input0_layout({batch, seq_length_q,  num_heads, head_size}, data_types::f16, format::bfyx);
    layout input1_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    layout input2_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    layout input3_layout({batch, num_heads, 1, seq_length_kv}, data_types::f16, format::bfyx);

    auto input0 = engine.allocate_memory(input0_layout);
    auto input1 = engine.allocate_memory(input1_layout);
    auto input2 = engine.allocate_memory(input2_layout);
    auto input3 = engine.allocate_memory(input3_layout);


    auto fill_random = [&](memory::ptr mem) {
        auto shp = mem->get_layout().get_shape();
        size_t sz = ov::shape_size(shp);
        auto data = rg.generate_random_1d<ov::float16>(sz, -1.0f, 1.0f);
        set_values(mem, data);
    };
    fill_random(input0);
    fill_random(input1);
    fill_random(input2);

    // attention mask with first position 0, all remaining positions -inf
    {
        size_t elems = batch * num_heads * 1 * seq_length_kv;
        std::vector<ov::float16> mask(elems);
        for (int kv = 0; kv < seq_length_kv; ++kv) {
            mask[kv] = kv == 0 ? ov::float16(0.0f) : ov::float16(-std::numeric_limits<float>::infinity());
        }
        set_values(input3, mask);
    }

    topology topology;
    topology.add(input_layout("input0", input0_layout));
    topology.add(input_layout("input1", input1_layout));
    topology.add(input_layout("input2", input2_layout));
    topology.add(input_layout("input3", input3_layout));
    topology.add(scaled_dot_product_attention("sdpa", {input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3")},
                                              false, -1, {0,2,1,3}, {0,2,1,3}, {0,2,1,3}, {0,1,2,3}, {}, false));
    topology.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"sdpa", {format::type::bfyx, "sdpa_opt"}}
    }));
    auto network = get_network(engine, topology, cfg, get_test_stream_ptr(), false);
    network->set_input_data("input0", input0);
    network->set_input_data("input1", input1);
    network->set_input_data("input2", input2);
    network->set_input_data("input3", input3);
    auto output = network->execute().at("result").get_memory();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_FALSE(std::isnan(static_cast<float>(output_ptr[i])));
    }

    // With only first KV valid, output should approximate value vector at KV index 0.
    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_ptr(input2, get_test_stream());
    for (int hs = 0; hs < head_size; ++hs) {
        ASSERT_NEAR(static_cast<float>(ref_ptr[hs]), static_cast<float>(output_ptr[hs]), 1e-2f);
    }
}

TEST(sdpa_gpu_custom, scalar_placeholder_mask_matches_scale_only) {
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int batch = 1;
    const int seq_length_q = 4;
    const int seq_length_kv = 6;
    const int num_heads = 2;
    const int head_size = 32;
    const float scale_val = 0.35f;

    const layout q_layout({batch, seq_length_q, num_heads, head_size}, data_types::f16, format::bfyx);
    const layout k_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    const layout v_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    const layout scalar_mask_layout{ov::PartialShape{}, data_types::f16, format::bfyx};

    auto q_mem = engine.allocate_memory(q_layout);
    auto k_mem = engine.allocate_memory(k_layout);
    auto v_mem = engine.allocate_memory(v_layout);
    auto scalar_mask_mem = engine.allocate_memory(scalar_mask_layout);

    auto fill_random = [&](const memory::ptr& mem) {
        const auto shape = mem->get_layout().get_shape();
        const size_t elements_num = ov::shape_size(shape);
        auto data = rg.generate_random_1d<ov::float16>(elements_num, -1.0f, 1.0f);
        set_values(mem, data);
    };

    fill_random(q_mem);
    fill_random(k_mem);
    fill_random(v_mem);
    set_values(scalar_mask_mem, {ov::float16(1.0f)});

    auto run_sdpa = [&](bool use_placeholder_mask) {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", v_layout));
        std::vector<input_info> inputs = {input_info("q"), input_info("k"), input_info("v")};
        if (use_placeholder_mask) {
            topo.add(input_layout("mask", scalar_mask_layout));
            inputs.push_back(input_info("mask"));
        }

        auto sdpa_prim = scaled_dot_product_attention("sdpa",
                                                      inputs,
                                                      false,
                                                      -1,
                                                      {0, 2, 1, 3},
                                                      {0, 2, 1, 3},
                                                      {0, 2, 1, 3},
                                                      {0, 1, 2, 3},
                                                      {},
                                                      false);
        sdpa_prim.scale_val = scale_val;

        topo.add(sdpa_prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"sdpa", {format::type::bfyx, "sdpa_opt"}}}));

        auto network = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        network->set_input_data("q", q_mem);
        network->set_input_data("k", k_mem);
        network->set_input_data("v", v_mem);
        if (use_placeholder_mask) {
            network->set_input_data("mask", scalar_mask_mem);
        }

        return network->execute().at("result").get_memory();
    };

    auto output_without_mask = run_sdpa(false);
    auto output_with_placeholder_mask = run_sdpa(true);

    cldnn::mem_lock<ov::float16, mem_lock_type::read> without_mask_ptr(output_without_mask, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> with_placeholder_mask_ptr(output_with_placeholder_mask, get_test_stream());

    ASSERT_EQ(without_mask_ptr.size(), with_placeholder_mask_ptr.size());
    for (size_t i = 0; i < without_mask_ptr.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(without_mask_ptr[i]), static_cast<float>(with_placeholder_mask_ptr[i]), 1e-3f)
            << "Mismatch at index " << i
            << ", Expected : " << static_cast<float>(without_mask_ptr[i])
            << " actual : " << static_cast<float>(with_placeholder_mask_ptr[i])
            << std::endl;
    }
}
} // namespace
