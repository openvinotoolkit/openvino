// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/rope.hpp>

using namespace cldnn;
using namespace ::tests;

// ============================================================================
// BF16/F16 RoPE reference (RotateHalf mode). For BF16 the kernel runs the
// VEC_SIZE==1 path with DECODE_INPUT0_COMPUTE_TYPE decode; for F16 it may run the
// vectorized path, but the math is the same half-rotation.
// Input layout [batch, head, seq, head_size] bfyx (or [batch, seq, head, head_size]
// when input_trans0213). Cos/Sin tables [1, 1, seq, head_size].
// Rotate pairs (i, i + rotary_ndims/2):
//   out[i]        = cos[p, i]        * in[i]        - sin[p, i]        * in[i + half]
//   out[i + half] = cos[p, i + half] * in[i + half] + sin[p, i + half] * in[i]
// (default COS_SIN_TABLE_OFFSET == rotary_ndims/2). Computed in FP32.
// ============================================================================
static void rope_ref(const memory::ptr input, const memory::ptr cos, const memory::ptr sin,
                     memory::ptr output, size_t batch, size_t head_cnt, size_t seq,
                     size_t head_size, size_t rotary_ndims, bool input_trans0213) {
    const size_t half = rotary_ndims / 2;

    cldnn::mem_lock<ov::bfloat16> src_bf16(input, get_test_stream());
    cldnn::mem_lock<ov::float16> src_f16(input, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> cosv_bf16(cos, get_test_stream());
    cldnn::mem_lock<ov::float16> cosv_f16(cos, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> sinv_bf16(sin, get_test_stream());
    cldnn::mem_lock<ov::float16> sinv_f16(sin, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> dst_bf16(output, get_test_stream());
    cldnn::mem_lock<ov::float16> dst_f16(output, get_test_stream());

    auto read_src = [&](size_t i) -> float {
        if (input->get_layout().data_type == data_types::bf16)
            return static_cast<float>(src_bf16[i]);
        else
            return static_cast<float>(src_f16[i]);
    };
    auto read_cos = [&](size_t i) -> float {
        if (cos->get_layout().data_type == data_types::bf16)
            return static_cast<float>(cosv_bf16[i]);
        else
            return static_cast<float>(cosv_f16[i]);
    };
    auto read_sin = [&](size_t i) -> float {
        if (sin->get_layout().data_type == data_types::bf16)
            return static_cast<float>(sinv_bf16[i]);
        else
            return static_cast<float>(sinv_f16[i]);
    };
    auto write_dst = [&](size_t i, float v) {
        if (output->get_layout().data_type == data_types::bf16)
            dst_bf16[i] = ov::bfloat16(v);
        else
            dst_f16[i] = ov::float16(v);
    };

    auto out_off = [&](size_t b, size_t h, size_t p, size_t x) {
        return b * head_cnt * seq * head_size + h * seq * head_size + p * head_size + x;
    };
    auto cos_off = [&](size_t p, size_t x) { return p * head_size + x; };

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < head_cnt; ++h) {
            for (size_t p = 0; p < seq; ++p) {
                size_t base_out = out_off(b, h, p, 0);
                size_t base_in;
                if (input_trans0213) {
                    // input [batch, seq, head, head_size] bfyx: b=b, f=seq, y=head, x=head_size
                    base_in = b * seq * head_cnt * head_size + p * head_cnt * head_size + h * head_size;
                } else {
                    base_in = base_out;
                }
                for (size_t i = 0; i < half; ++i) {
                    float in1 = read_src(base_in + i);
                    float in2 = read_src(base_in + half + i);
                    float c1 = read_cos(cos_off(p, i));
                    float s1 = read_sin(cos_off(p, i));
                    float c2 = read_cos(cos_off(p, half + i));
                    float s2 = read_sin(cos_off(p, half + i));
                    write_dst(base_out + i, c1 * in1 - s1 * in2);
                    write_dst(base_out + half + i, c2 * in2 + s2 * in1);
                }
            }
        }
    }
}

// ============================================================================
// BF16/F16 RoPE reference (interleaved mode, config.is_interleaved). Cos/Sin are
// per-element tables with the same layout as input/output
// [batch, head, seq, head_size] bfyx. Rotate adjacent pairs (i, i + 1):
//   out[i]   = cos[i]   * in[i]   - sin[i]   * in[i + 1]
//   out[i+1] = cos[i+1] * in[i+1] + sin[i+1] * in[i]
// Elements at/above rotary_ndims are copied as-is. Computed in FP32.
// ============================================================================
static void rope_interleaved_ref(const memory::ptr input, const memory::ptr cos, const memory::ptr sin,
                                 memory::ptr output, size_t batch, size_t head_cnt, size_t seq,
                                 size_t head_size, size_t rotary_ndims) {
    cldnn::mem_lock<ov::bfloat16> src_bf16(input, get_test_stream());
    cldnn::mem_lock<ov::float16> src_f16(input, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> cosv_bf16(cos, get_test_stream());
    cldnn::mem_lock<ov::float16> cosv_f16(cos, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> sinv_bf16(sin, get_test_stream());
    cldnn::mem_lock<ov::float16> sinv_f16(sin, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> dst_bf16(output, get_test_stream());
    cldnn::mem_lock<ov::float16> dst_f16(output, get_test_stream());

    auto read_src = [&](size_t i) -> float {
        if (input->get_layout().data_type == data_types::bf16)
            return static_cast<float>(src_bf16[i]);
        else
            return static_cast<float>(src_f16[i]);
    };
    auto read_cos = [&](size_t i) -> float {
        if (cos->get_layout().data_type == data_types::bf16)
            return static_cast<float>(cosv_bf16[i]);
        else
            return static_cast<float>(cosv_f16[i]);
    };
    auto read_sin = [&](size_t i) -> float {
        if (sin->get_layout().data_type == data_types::bf16)
            return static_cast<float>(sinv_bf16[i]);
        else
            return static_cast<float>(sinv_f16[i]);
    };
    auto write_dst = [&](size_t i, float v) {
        if (output->get_layout().data_type == data_types::bf16)
            dst_bf16[i] = ov::bfloat16(v);
        else
            dst_f16[i] = ov::float16(v);
    };

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < head_cnt; ++h) {
            for (size_t p = 0; p < seq; ++p) {
                const size_t base = b * head_cnt * seq * head_size + h * seq * head_size + p * head_size;
                for (size_t i = 0; i < rotary_ndims; i += 2) {
                    float in1 = read_src(base + i);
                    float in2 = read_src(base + i + 1);
                    write_dst(base + i, read_cos(base + i) * in1 - read_sin(base + i) * in2);
                    write_dst(base + i + 1, read_cos(base + i + 1) * in2 + read_sin(base + i + 1) * in1);
                }
                for (size_t i = rotary_ndims; i < head_size; ++i) {
                    write_dst(base + i, read_src(base + i));
                }
            }
        }
    }
}

// ============================================================================
// BF16/F16 RoPE reference (QWEN per-head mode, config.is_qwen). Input
// [batch, seq, head_cnt*head_size] bfyx, per-head cos/sin tables
// [batch, seq, head, head_size], output [batch, seq, head, head_size].
// Same half-offset rotation as RotateHalf, but the tables are indexed per head:
//   out[h][r]        = cos[h][r]        * in[h][r]        - sin[h][r]        * in[h][r + half]
//   out[h][r + half] = cos[h][r + half] * in[h][r + half] + sin[h][r + half] * in[h][r]
// Elements at/above rotary_ndims are copied as-is. Computed in FP32.
// ============================================================================
static void rope_qwen_ref(const memory::ptr input, const memory::ptr cos, const memory::ptr sin,
                          memory::ptr output, size_t batch, size_t head_cnt, size_t seq,
                          size_t head_size, size_t rotary_ndims) {
    const size_t half = rotary_ndims / 2;

    cldnn::mem_lock<ov::bfloat16> src_bf16(input, get_test_stream());
    cldnn::mem_lock<ov::float16> src_f16(input, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> cosv_bf16(cos, get_test_stream());
    cldnn::mem_lock<ov::float16> cosv_f16(cos, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> sinv_bf16(sin, get_test_stream());
    cldnn::mem_lock<ov::float16> sinv_f16(sin, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> dst_bf16(output, get_test_stream());
    cldnn::mem_lock<ov::float16> dst_f16(output, get_test_stream());

    auto read_src = [&](size_t i) -> float {
        if (input->get_layout().data_type == data_types::bf16)
            return static_cast<float>(src_bf16[i]);
        else
            return static_cast<float>(src_f16[i]);
    };
    auto read_cos = [&](size_t i) -> float {
        if (cos->get_layout().data_type == data_types::bf16)
            return static_cast<float>(cosv_bf16[i]);
        else
            return static_cast<float>(cosv_f16[i]);
    };
    auto read_sin = [&](size_t i) -> float {
        if (sin->get_layout().data_type == data_types::bf16)
            return static_cast<float>(sinv_bf16[i]);
        else
            return static_cast<float>(sinv_f16[i]);
    };
    auto write_dst = [&](size_t i, float v) {
        if (output->get_layout().data_type == data_types::bf16)
            dst_bf16[i] = ov::bfloat16(v);
        else
            dst_f16[i] = ov::float16(v);
    };

    for (size_t b = 0; b < batch; ++b) {
        for (size_t p = 0; p < seq; ++p) {
            for (size_t h = 0; h < head_cnt; ++h) {
                // input row, table row and output row share the same linear base
                const size_t base = b * seq * head_cnt * head_size + p * head_cnt * head_size + h * head_size;
                for (size_t i = 0; i < half; ++i) {
                    float in1 = read_src(base + i);
                    float in2 = read_src(base + half + i);
                    float c1 = read_cos(base + i);
                    float s1 = read_sin(base + i);
                    float c2 = read_cos(base + half + i);
                    float s2 = read_sin(base + half + i);
                    write_dst(base + i, c1 * in1 - s1 * in2);
                    write_dst(base + half + i, c2 * in2 + s2 * in1);
                }
                for (size_t i = rotary_ndims; i < head_size; ++i) {
                    write_dst(base + i, read_src(base + i));
                }
            }
        }
    }
}

// ============================================================================
// BF16/F16 RoPE reference (ChatGLM mode, config.is_chatglm). Input
// [seq, batch, head_cnt*head_size] bfyx, output [seq, batch, head, head_size].
// Rotate adjacent pairs (r, r+1):
//   cache:     interleaved cos_sin table [seq, batch, 1, rotary_ndims]
//              (cos at even, sin at odd indices)
//   non-cache: per-pair cos/sin tables [seq, batch, 1, rotary_ndims/2]
//   out[r]   = cos * in[r]   - sin * in[r+1]
//   out[r+1] = sin * in[r]   + cos * in[r+1]
// Computed in FP32.
// ============================================================================
static void rope_chatglm_ref(const memory::ptr input, const memory::ptr cos, const memory::ptr sin,
                             memory::ptr output, size_t batch, size_t seq, size_t head_cnt,
                             size_t head_size, size_t rotary_ndims, bool use_rope_cache) {
    const size_t half = rotary_ndims / 2;

    cldnn::mem_lock<ov::bfloat16> src_bf16(input, get_test_stream());
    cldnn::mem_lock<ov::float16> src_f16(input, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> cosv_bf16(cos, get_test_stream());
    cldnn::mem_lock<ov::float16> cosv_f16(cos, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> sinv_bf16(sin, get_test_stream());
    cldnn::mem_lock<ov::float16> sinv_f16(sin, get_test_stream());
    cldnn::mem_lock<ov::bfloat16> dst_bf16(output, get_test_stream());
    cldnn::mem_lock<ov::float16> dst_f16(output, get_test_stream());

    auto read_src = [&](size_t i) -> float {
        if (input->get_layout().data_type == data_types::bf16)
            return static_cast<float>(src_bf16[i]);
        else
            return static_cast<float>(src_f16[i]);
    };
    auto read_cos = [&](size_t i) -> float {
        if (cos->get_layout().data_type == data_types::bf16)
            return static_cast<float>(cosv_bf16[i]);
        else
            return static_cast<float>(cosv_f16[i]);
    };
    auto read_sin = [&](size_t i) -> float {
        if (sin->get_layout().data_type == data_types::bf16)
            return static_cast<float>(sinv_bf16[i]);
        else
            return static_cast<float>(sinv_f16[i]);
    };
    auto write_dst = [&](size_t i, float v) {
        if (output->get_layout().data_type == data_types::bf16)
            dst_bf16[i] = ov::bfloat16(v);
        else
            dst_f16[i] = ov::float16(v);
    };

    for (size_t p = 0; p < seq; ++p) {
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < head_cnt; ++h) {
                // input and output rows share the same linear base
                const size_t base = p * batch * head_cnt * head_size + b * head_cnt * head_size + h * head_size;
                for (size_t rf = 0; rf < half; ++rf) {
                    const size_t r = rf * 2;
                    float in1 = read_src(base + r);
                    float in2 = read_src(base + r + 1);
                    float c, s;
                    if (use_rope_cache) {
                        // interleaved table: cos at even, sin at odd
                        const size_t tb = p * batch * rotary_ndims + b * rotary_ndims;
                        c = read_cos(tb + r);
                        s = read_cos(tb + r + 1);
                    } else {
                        // per-pair tables
                        const size_t tb = p * batch * half + b * half;
                        c = read_cos(tb + rf);
                        s = read_sin(tb + rf);
                    }
                    write_dst(base + r, c * in1 - s * in2);
                    write_dst(base + r + 1, s * in1 + c * in2);
                }
            }
        }
    }
}

enum class rope_variant { rotate_half, interleaved, qwen_per_head, chatglm };

static void run_rope(data_types dt, const ov::PartialShape& in_shape, size_t head_cnt, size_t head_size,
                     bool input_trans0213, bool dynamic,
                     rope_variant variant = rope_variant::rotate_half,
                     bool use_rope_cache = false) {
    auto& engine = get_test_engine();
    size_t batch = in_shape[0].get_length();
    size_t seq;
    if (variant == rope_variant::chatglm) {
        // chatglm input is [seq, batch, head_cnt*head_size] (seq is the leading dim)
        seq = in_shape[0].get_length();
        batch = in_shape[1].get_length();
    } else if (variant == rope_variant::qwen_per_head || input_trans0213) {
        seq = in_shape[1].get_length();
    } else {
        seq = in_shape[2].get_length();
    }
    const size_t rotary_ndims = head_size;  // All tests cover full head_size

    // Variant-specific cos/sin table and output shapes (bfyx)
    ov::PartialShape cos_shape;
    ov::PartialShape out_shape;
    if (variant == rope_variant::chatglm) {
        // non-cache: per-pair cos/sin [seq, batch, 1, rotary_ndims/2]
        // cache: interleaved cos_sin [seq, batch, 1, rotary_ndims]
        cos_shape = ov::PartialShape{static_cast<int64_t>(seq), static_cast<int64_t>(batch), 1,
                                     static_cast<int64_t>(use_rope_cache ? rotary_ndims : rotary_ndims / 2)};
        out_shape = ov::PartialShape{static_cast<int64_t>(seq), static_cast<int64_t>(batch),
                                     static_cast<int64_t>(head_cnt), static_cast<int64_t>(head_size)};
    } else if (variant == rope_variant::qwen_per_head) {
        // input [batch, seq, head_cnt*head_size] -> per-head tables [batch, seq, head, head_size]
        cos_shape = ov::PartialShape{static_cast<int64_t>(batch), static_cast<int64_t>(seq),
                                     static_cast<int64_t>(head_cnt), static_cast<int64_t>(head_size)};
        out_shape = cos_shape;
    } else if (variant == rope_variant::interleaved) {
        // per-element tables with the same layout as the input
        cos_shape = in_shape;
        out_shape = in_shape;
    } else {
        cos_shape = ov::PartialShape{1, 1, static_cast<int64_t>(seq), static_cast<int64_t>(head_size)};
        // output is [batch, head, seq, head_size] (transpose swaps input dims 1&2)
        out_shape = input_trans0213
            ? ov::PartialShape{static_cast<int64_t>(batch), static_cast<int64_t>(head_cnt),
                               static_cast<int64_t>(seq), static_cast<int64_t>(head_size)}
            : in_shape;
    }

    auto input = engine.allocate_memory({in_shape, dt, format::bfyx});
    auto cos = engine.allocate_memory({cos_shape, dt, format::bfyx});
    auto sin = engine.allocate_memory({cos_shape, dt, format::bfyx});
    auto output_ref = engine.allocate_memory({out_shape, dt, format::bfyx});

    if (dt == data_types::bf16) {
        tests::set_random_values<ov::bfloat16>(input, true, 7, 4);
        tests::set_random_values<ov::bfloat16>(cos, true, 7, 1);
        tests::set_random_values<ov::bfloat16>(sin, true, 7, 1);
    } else {
        tests::set_random_values<ov::float16>(input, true, 10, 4);
        tests::set_random_values<ov::float16>(cos, true, 10, 1);
        tests::set_random_values<ov::float16>(sin, true, 10, 1);
    }

    if (variant == rope_variant::interleaved) {
        rope_interleaved_ref(input, cos, sin, output_ref, batch, head_cnt, seq, head_size, rotary_ndims);
    } else if (variant == rope_variant::qwen_per_head) {
        rope_qwen_ref(input, cos, sin, output_ref, batch, head_cnt, seq, head_size, rotary_ndims);
    } else if (variant == rope_variant::chatglm) {
        rope_chatglm_ref(input, cos, sin, output_ref, batch, seq, head_cnt, head_size, rotary_ndims, use_rope_cache);
    } else {
        rope_ref(input, cos, sin, output_ref, batch, head_cnt, seq, head_size, rotary_ndims, input_trans0213);
    }

    ov::op::internal::RoPE::Config config;
    config.head_cnt = head_cnt;
    config.head_size = head_size;
    config.rotary_ndims = rotary_ndims;
    config.input_trans0213 = input_trans0213 && variant == rope_variant::rotate_half;
    config.is_interleaved = variant == rope_variant::interleaved;
    config.is_qwen = variant == rope_variant::qwen_per_head;
    config.is_chatglm = variant == rope_variant::chatglm;
    config.use_rope_cache = use_rope_cache;

    topology topology;
    auto in_lay = dynamic
        ? layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(),
                                  ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                 dt, format::bfyx}
        : input->get_layout();
    topology.add(input_layout("input", in_lay));
    if (variant == rope_variant::chatglm && use_rope_cache) {
        topology.add(input_layout("cos_sin", cos->get_layout()));
        topology.add(rope("rope", {input_info("input"), input_info("cos_sin")}, config));
    } else {
        topology.add(input_layout("cos", cos->get_layout()));
        topology.add(input_layout("sin", sin->get_layout()));
        topology.add(rope("rope", {input_info("input"), input_info("cos"), input_info("sin")}, config));
    }

    ExecutionConfig cfg = get_test_default_config(engine);
    if (dynamic)
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, cfg);
    network.set_input_data("input", input);
    if (variant == rope_variant::chatglm && use_rope_cache) {
        network.set_input_data("cos_sin", cos);
    } else {
        network.set_input_data("cos", cos);
        network.set_input_data("sin", sin);
    }

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "rope");

    auto output = outputs.at("rope").get_memory();
    ASSERT_EQ(output->get_layout().get_shape(), out_shape.get_shape());

    auto compare = [&](auto&& read_gpu) {
        cldnn::mem_lock<ov::bfloat16> ref_bf16(output_ref, get_test_stream());
        cldnn::mem_lock<ov::float16> ref_f16(output_ref, get_test_stream());
        // BF16 ~7 mantissa bits, F16 ~10; F16 accumulates in half on GPU
        float abs_floor = 0.02f;
        float rel_tol = (dt == data_types::bf16) ? 0.02f : 0.04f;
        for (size_t i = 0; i < output_ref->count(); ++i) {
            float gpu_val = read_gpu(i);
            float ref_val = (dt == data_types::bf16) ? static_cast<float>(ref_bf16[i])
                                                     : static_cast<float>(ref_f16[i]);
            float diff = std::abs(gpu_val - ref_val);
            float tolerance = std::max(abs_floor, std::abs(ref_val) * rel_tol);
            ASSERT_LE(diff, tolerance) << "Mismatch at i=" << i
                << " gpu=" << gpu_val << " ref=" << ref_val << " diff=" << diff;
        }
    };

    if (dt == data_types::bf16) {
        cldnn::mem_lock<ov::bfloat16> out_ptr(output, get_test_stream());
        compare([&](size_t i) { return static_cast<float>(out_ptr[i]); });
    } else {
        cldnn::mem_lock<ov::float16> out_ptr(output, get_test_stream());
        compare([&](size_t i) { return static_cast<float>(out_ptr[i]); });
    }
}

class rope_gpu_test : public ::testing::TestWithParam<data_types> {};

static std::string rope_test_name(testing::TestParamInfo<data_types> info) {
    return info.param == data_types::bf16 ? "bf16" : "f16";
}

INSTANTIATE_TEST_SUITE_P(smoke, rope_gpu_test,
                         ::testing::Values(data_types::bf16, data_types::f16),
                         rope_test_name);

// RoPE basic RotateHalf, 2 batches x 2 heads x 4 positions x 8 dims
TEST_P(rope_gpu_test, rope_basic) {
    run_rope(GetParam(), ov::PartialShape{2, 2, 4, 8}, 2, 8, false, false);
}

// RoPE input_trans0213 (input [b, seq, head, HS], output [b, head, seq, HS])
TEST_P(rope_gpu_test, rope_transpose_input) {
    run_rope(GetParam(), ov::PartialShape{2, 4, 2, 8}, 2, 8, true, false);
}

// RoPE larger realistic shape (8 heads, 16 positions, 64 dims) - exercises the F16
// vectorized (VEC_SIZE=16) kernel path
TEST_P(rope_gpu_test, rope_large) {
    run_rope(GetParam(), ov::PartialShape{1, 8, 16, 64}, 8, 64, false, false);
}

// RoPE dynamic shape (seq is dynamic)
TEST_P(rope_gpu_test, rope_dynamic) {
    run_rope(GetParam(), ov::PartialShape{1, 2, 4, 8}, 2, 8, false, true);
}

// RoPE interleaved variant (per-element cos/sin tables), small head - VEC_SIZE=1 path
TEST_P(rope_gpu_test, rope_interleaved) {
    run_rope(GetParam(), ov::PartialShape{2, 2, 4, 8}, 2, 8, false, false, rope_variant::interleaved);
}

// RoPE interleaved variant, 8 heads x 16 positions x 64 dims - VEC_SIZE=16 path
TEST_P(rope_gpu_test, rope_interleaved_large) {
    run_rope(GetParam(), ov::PartialShape{1, 8, 16, 64}, 8, 64, false, false, rope_variant::interleaved);
}

// RoPE QWEN per-head variant (input [b, seq, H*HS], per-head cos/sin tables), small head - VEC_SIZE=1 path
TEST_P(rope_gpu_test, rope_qwen) {
    run_rope(GetParam(), ov::PartialShape{2, 4, 16}, 2, 8, false, false, rope_variant::qwen_per_head);
}

// RoPE QWEN per-head variant, 8 heads x 16 positions x 64 dims - VEC_SIZE=16 path
TEST_P(rope_gpu_test, rope_qwen_large) {
    run_rope(GetParam(), ov::PartialShape{1, 16, 512}, 8, 64, false, false, rope_variant::qwen_per_head);
}

// RoPE ChatGLM variant (non-cache per-pair cos/sin, input [seq, batch, H*HS]), small head - VEC_SIZE=1
TEST_P(rope_gpu_test, rope_chatglm) {
    run_rope(GetParam(), ov::PartialShape{4, 2, 16}, 2, 8, false, false, rope_variant::chatglm);
}

// RoPE ChatGLM variant, 8 heads x 64 dims - VEC_SIZE=16 path (UNPACK_BF16_VEC, scalar
// TO_OUTPUT_TYPE write loop, non-cache ushort8 cos/sin read)
TEST_P(rope_gpu_test, rope_chatglm_large) {
    run_rope(GetParam(), ov::PartialShape{16, 1, 512}, 8, 64, false, false, rope_variant::chatglm);
}

// RoPE ChatGLM variant with use_rope_cache (interleaved cos_sin table, 2 inputs), small head - VEC_SIZE=1
TEST_P(rope_gpu_test, rope_chatglm_cache) {
    run_rope(GetParam(), ov::PartialShape{4, 2, 16}, 2, 8, false, false, rope_variant::chatglm, true);
}

// RoPE ChatGLM variant with use_rope_cache, 8 heads x 64 dims - VEC_SIZE=16 path
TEST_P(rope_gpu_test, rope_chatglm_cache_large) {
    run_rope(GetParam(), ov::PartialShape{16, 1, 512}, 8, 64, false, false, rope_variant::chatglm, true);
}
