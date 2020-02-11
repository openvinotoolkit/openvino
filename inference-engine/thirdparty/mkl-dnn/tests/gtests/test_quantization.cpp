/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <mkldnn_types.h>
#include "gtest/gtest.h"
#include "mkldnn_test_common.hpp"
#include "mkldnn.hpp"

namespace mkldnn {

struct quantization_test_params {
    engine::kind engine_kind;
    algorithm alg_kind;
    memory::format data_format;
    int axis;
    memory::dims dims;
    int levels;
};

template <typename src_data_t, typename dst_data_t>
void check_quantization_fwd(const quantization_test_params &p,
        const memory::desc &src_md, const memory &src,
        const memory &input_low, const memory &input_high, const memory &output_low, const memory &output_high,
        const memory &dst) {
    auto src_data = (src_data_t*)src.get_data_handle();
    auto input_low_data = (float*)input_low.get_data_handle();
    auto input_high_data = (float*)input_high.get_data_handle();
    auto output_low_data = (float*)output_low.get_data_handle();
    auto output_high_data = (float*)output_high.get_data_handle();
    auto dst_data = (dst_data_t*)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc input_low_d = input_low.get_primitive_desc().desc();
    const memory::desc input_high_d = input_high.get_primitive_desc().desc();
    const memory::desc output_low_d = output_low.get_primitive_desc().desc();
    const memory::desc output_high_d = output_high.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    int N = src_md.data.dims[0];
    int C = src_md.data.dims[1];
    int D = src_md.data.ndims == 5 ? src_md.data.dims[2] : 1;
    int H = src_md.data.ndims == 3 ? src_md.data.dims[2] : src_md.data.ndims > 2 ? src_md.data.dims[src_md.data.ndims - 2] : 1;
    int W = src_md.data.ndims > 3 ? src_md.data.dims[src_md.data.ndims - 1] : 1;

    int padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];
    int padded_oc = dst_d.data.layout_desc.blocking.padding_dims[1];

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int d = 0; d < D; ++d) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int src_idx = n * padded_ic * D * H * W + c * D * H * W + d * H * W + h * W + w;
                        int wei_idx = p.axis == 0 ? n : c;

                        src_data_t s_val = src_data[map_index(src_d, src_idx)];

                        float in_low = input_low_data[map_index(input_low_d, wei_idx)];
                        float in_high = input_high_data[map_index(input_high_d, wei_idx)];
                        float out_low = output_low_data[map_index(output_low_d, wei_idx)];
                        float out_high = output_high_data[map_index(output_high_d, wei_idx)];

                        float tmp_val = 0;
                        if (s_val <= in_low)
                            tmp_val = out_low;
                        else if (s_val > in_high)
                            tmp_val = out_high;
                        else
                            tmp_val = roundf((s_val - in_low) / (in_high - in_low) * (p.levels - 1)) /
                                      (p.levels - 1) * (out_high - out_low) + out_low;

                        int dst_idx = n * padded_oc * D * H * W + c * D * H * W + d * H * W + h * W + w;

                        if (data_traits<dst_data_t>::data_type == memory::data_type::f32) {
                            dst_data_t dst_val = tmp_val;

                            EXPECT_NEAR(dst_data[map_index(dst_d, dst_idx)], dst_val, 1e-4);
                        } else {
                            dst_data_t dst_val = saturate<dst_data_t>(tmp_val);

                            EXPECT_NEAR(dst_data[map_index(dst_d, dst_idx)], dst_val, 1);
                        }
                    }
                }
            }
        }
    }
}

template <typename src_data_t, typename dst_data_t>
class quantization_test : public ::testing::TestWithParam<quantization_test_params> {
private:

protected:
    virtual void SetUp() {
        auto p = ::testing::TestWithParam<quantization_test_params>::GetParam();

        ASSERT_TRUE(p.axis == 0 || p.axis == 1);

        auto eng = engine(p.engine_kind, 0);
        auto src_data_type = data_traits<src_data_t>::data_type;
        auto wei_data_type = data_traits<float>::data_type;
        auto dst_data_type = data_traits<dst_data_t>::data_type;

        memory::dims src_dims = p.dims;
        memory::dims wei_dims = p.axis == 0 ? memory::dims({src_dims[0]}) : memory::dims({src_dims[1]});
        memory::dims dst_dims = p.dims;

        auto src_desc = create_md(src_dims, src_data_type, p.data_format);
        auto dst_desc = create_md(dst_dims, dst_data_type, p.data_format);

        auto input_low_desc = create_md(wei_dims, wei_data_type, memory::format::x);
        auto input_high_desc = create_md(wei_dims, wei_data_type, memory::format::x);
        auto output_low_desc = create_md(wei_dims, wei_data_type, memory::format::x);
        auto output_high_desc = create_md(wei_dims, wei_data_type, memory::format::x);

        auto crop_low_desc = create_md(wei_dims, wei_data_type, memory::format::x);
        auto crop_high_desc = create_md(wei_dims, wei_data_type, memory::format::x);
        auto input_scale_desc = create_md(wei_dims, wei_data_type, memory::format::x);
        auto input_shift_desc = create_md(wei_dims, wei_data_type, memory::format::x);
        auto output_scale_desc = create_md(wei_dims, wei_data_type, memory::format::x);
        auto output_shift_desc = create_md(wei_dims, wei_data_type, memory::format::x);

        auto src = test_memory(src_desc, eng);
        auto dst = test_memory(dst_desc, eng);

        auto input_low = test_memory(input_low_desc, eng);
        auto input_high = test_memory(input_high_desc, eng);
        auto output_low = test_memory(output_low_desc, eng);
        auto output_high = test_memory(output_high_desc, eng);

        auto crop_low = test_memory(crop_low_desc, eng);
        auto crop_high = test_memory(crop_high_desc, eng);
        auto input_scale = test_memory(input_scale_desc, eng);
        auto input_shift = test_memory(input_shift_desc, eng);
        auto output_scale = test_memory(output_scale_desc, eng);
        auto output_shift = test_memory(output_shift_desc, eng);

        float min_val = -500.f;
        float max_val = 700.f;
        float low_min_val = -60.f;
        float low_max_val = -40.f;
        float high_min_val = 40.f;
        float high_max_val = 60.f;

        fill_data<src_data_t>(src.get_size() / sizeof(src_data_t), (src_data_t *)src.get().get_data_handle(), min_val, max_val);

        fill_data<float>(input_low.get_size() / sizeof(float), (float*)input_low.get().get_data_handle(), low_min_val, low_max_val);
        fill_data<float>(input_high.get_size() / sizeof(float), (float*)input_high.get().get_data_handle(), high_min_val, high_max_val);
        fill_data<float>(output_low.get_size() / sizeof(float), (float*)output_low.get().get_data_handle(), low_min_val, low_max_val);
        fill_data<float>(output_high.get_size() / sizeof(float), (float*)output_high.get().get_data_handle(), high_min_val, high_max_val);

        float* p_input_low = (float*)input_low.get().get_data_handle();
        float* p_input_high = (float*)input_high.get().get_data_handle();
        float* p_output_low = (float*)output_low.get().get_data_handle();
        float* p_output_high = (float*)output_high.get().get_data_handle();

        float* p_crop_low = (float*)crop_low.get().get_data_handle();
        float* p_crop_high = (float*)crop_high.get().get_data_handle();
        float* p_input_scale = (float*)input_scale.get().get_data_handle();
        float* p_input_shift = (float*)input_shift.get().get_data_handle();
        float* p_output_scale = (float*)output_scale.get().get_data_handle();
        float* p_output_shift = (float*)output_shift.get().get_data_handle();

        for (int i = 0; i < wei_dims[0]; i++) {
            p_crop_low[i] = p_input_low[i];
            p_crop_high[i] = p_input_high[i];

            p_input_scale[i] = ((float)p.levels - 1) / (p_input_high[i] - p_input_low[i]);
            p_input_shift[i] =  -(p_input_low[i] * ((float)p.levels - 1)) / (p_input_high[i] - p_input_low[i]);
            p_output_scale[i] = (p_output_high[i] - p_output_low[i]) / (p.levels - 1);
            p_output_shift[i] = p_output_low[i];
        }

        std::vector<primitive> pipeline;
        auto quantization_desc = quantization_forward::desc(prop_kind::forward_training, p.alg_kind, p.axis,
                src_desc, crop_low_desc, crop_high_desc, input_scale_desc, input_shift_desc, output_scale_desc, output_shift_desc, dst_desc);
        auto quantization_prim_desc = quantization_forward::primitive_desc(quantization_desc, eng);
        auto quantization = quantization_forward(quantization_prim_desc,
                src.get(), crop_low.get(), crop_high.get(), input_scale.get(), input_shift.get(), output_scale.get(), output_shift.get(), dst.get());

        pipeline.push_back(quantization);
        auto s = stream(stream::kind::lazy);
        s.submit(pipeline).wait();

        check_quantization_fwd<src_data_t,dst_data_t>(p, src_desc, src.get(), input_low.get(), input_high.get(), output_low.get(), output_high.get(), dst.get());
    }
};

using quantization_test_f32f32 = quantization_test<float, float>;
using quantization_test_f32u8 = quantization_test<float, uint8_t>;
using quantization_test_f32s8 = quantization_test<float, int8_t>;
using quantization_test_u8u8 = quantization_test<uint8_t, uint8_t>;

#define EXPAND(args) args

#define EXPAND_FORMATS(data) memory::format::data

#define ENGINE engine::kind::cpu

#define PARAMS_2D(alg, data, axis, mb, c, levels) \
    quantization_test_params { ENGINE, algorithm::alg, \
    EXPAND_FORMATS(data), axis, {mb, c}, levels}

#define PARAMS_3D(alg, data, axis, mb, c, h, levels) \
    quantization_test_params { ENGINE, algorithm::alg, \
    EXPAND_FORMATS(data), axis, {mb, c, h}, levels}

#define PARAMS_4D(alg, data, axis, mb, c, h, w, levels) \
    quantization_test_params { ENGINE, algorithm::alg, \
    EXPAND_FORMATS(data), axis, {mb, c, h, w}, levels}

#define PARAMS_5D(alg, data, axis, mb, c, d, h, w, levels) \
    quantization_test_params { ENGINE, algorithm::alg, \
    EXPAND_FORMATS(data), axis, {mb, c, d, h, w}, levels}

#define PARAMS_ALL_ALG_2D(...) \
    EXPAND(PARAMS_2D(quantization_quantize_dequantize, __VA_ARGS__))

#define PARAMS_ALL_ALG_3D(...) \
    EXPAND(PARAMS_3D(quantization_quantize_dequantize, __VA_ARGS__))

#define PARAMS_ALL_ALG_4D(...) \
    EXPAND(PARAMS_4D(quantization_quantize_dequantize, __VA_ARGS__))

#define PARAMS_ALL_ALG_5D(...) \
    EXPAND(PARAMS_5D(quantization_quantize_dequantize, __VA_ARGS__))

#define INST_TEST_CASE(str, test, ...) INSTANTIATE_TEST_CASE_P( \
        str, test, ::testing::Values(__VA_ARGS__))

#define TEST_SIZES_2D(format) \
    PARAMS_ALL_ALG_2D(format, 0, 1, 8, 255), \
    PARAMS_ALL_ALG_2D(format, 1, 1, 8, 255), \
    PARAMS_ALL_ALG_2D(format, 0, 10, 10,255), \
    PARAMS_ALL_ALG_2D(format, 0, 128, 32, 255), \
    PARAMS_ALL_ALG_2D(format, 1, 10, 10, 255), \
    PARAMS_ALL_ALG_2D(format, 1, 128, 32, 255)

#define TEST_SIZES_3D(format) \
    PARAMS_ALL_ALG_3D(format, 0, 1, 8, 7, 255), \
    PARAMS_ALL_ALG_3D(format, 1, 1, 8, 7, 255), \
    PARAMS_ALL_ALG_3D(format, 0, 10, 10, 1, 255), \
    PARAMS_ALL_ALG_3D(format, 0, 128, 32, 3, 255), \
    PARAMS_ALL_ALG_3D(format, 1, 10, 10, 4, 255), \
    PARAMS_ALL_ALG_3D(format, 1, 128, 32, 13, 255)

#define TEST_SIZES_4D(format) \
    PARAMS_ALL_ALG_4D(format, 0, 1, 8, 4, 4, 255), \
    PARAMS_ALL_ALG_4D(format, 0, 2, 16, 10, 8, 255), \
    PARAMS_ALL_ALG_4D(format, 0, 10, 10, 10, 10, 255), \
    PARAMS_ALL_ALG_4D(format, 0, 128, 32, 8, 16, 255), \
    PARAMS_ALL_ALG_4D(format, 0, 1, 1, 1, 1, 255), \
    PARAMS_ALL_ALG_4D(format, 0, 3, 5, 7, 11, 255), \
    PARAMS_ALL_ALG_4D(format, 0, 2, 333, 8, 3, 255), \
    PARAMS_ALL_ALG_4D(format, 1, 1, 8, 4, 4, 255), \
    PARAMS_ALL_ALG_4D(format, 1, 2, 16, 8, 8, 255), \
    PARAMS_ALL_ALG_4D(format, 1, 2, 16, 10, 8, 255), \
    PARAMS_ALL_ALG_4D(format, 1, 10, 10, 10, 10, 255), \
    PARAMS_ALL_ALG_4D(format, 1, 128, 32, 8, 16, 255), \
    PARAMS_ALL_ALG_4D(format, 1, 1, 1, 1, 1, 255), \
    PARAMS_ALL_ALG_4D(format, 1, 3, 5, 7, 11, 255), \
    PARAMS_ALL_ALG_4D(format, 1, 2, 333, 8, 3, 255)

#define TEST_SIZES_5D(format) \
    PARAMS_ALL_ALG_5D(format, 0, 1, 8, 4, 4, 4, 255), \
    PARAMS_ALL_ALG_5D(format, 1, 1, 8, 4, 4, 4, 255), \
    PARAMS_ALL_ALG_5D(format, 0, 10, 10, 10, 10, 10, 255), \
    PARAMS_ALL_ALG_5D(format, 0, 128, 32, 4, 8, 16, 255), \
    PARAMS_ALL_ALG_5D(format, 1, 10, 10, 10, 10, 10, 255), \
    PARAMS_ALL_ALG_5D(format, 1, 128, 32, 4, 8, 16, 255)

#define INST_TEST_CASE_PLANAR_2D_P(test) \
TEST_P(test, TestsQuantizationPlanar2D) {} \
INST_TEST_CASE(SimplePlanar2D, test, \
     TEST_SIZES_2D(nc) \
);

#define INST_TEST_CASE_PLANAR_3D_P(test) \
TEST_P(test, TestsQuantizationPlanar3D) {} \
INST_TEST_CASE(SimplePlanar3D, test, \
     TEST_SIZES_3D(tnc) \
);

#define INST_TEST_CASE_BLOCKED_4D_P(test) \
TEST_P(test, TestsQuantizationBlocked4D) {} \
INST_TEST_CASE(SimpleBlocked4D, test, \
     TEST_SIZES_4D(nChw8c), \
     TEST_SIZES_4D(nChw16c) \
);

#define INST_TEST_CASE_PLANAR_4D_P(test) \
TEST_P(test, TestsQuantizationPlanar4D) {} \
INST_TEST_CASE(SimplePlanar4D, test, \
     TEST_SIZES_4D(nhwc), \
     TEST_SIZES_4D(nchw) \
);

#define INST_TEST_CASE_PLANAR_5D_P(test) \
TEST_P(test, TestsQuantizationPlanar5D) {} \
INST_TEST_CASE(SimplePlanar5D, test, \
     TEST_SIZES_5D(ndhwc), \
     TEST_SIZES_5D(ncdhw) \
);

INST_TEST_CASE_PLANAR_2D_P(quantization_test_f32f32)
INST_TEST_CASE_PLANAR_2D_P(quantization_test_f32u8)
INST_TEST_CASE_PLANAR_2D_P(quantization_test_f32s8)
INST_TEST_CASE_PLANAR_2D_P(quantization_test_u8u8)

INST_TEST_CASE_PLANAR_3D_P(quantization_test_f32f32)
INST_TEST_CASE_PLANAR_3D_P(quantization_test_f32u8)
INST_TEST_CASE_PLANAR_3D_P(quantization_test_f32s8)
INST_TEST_CASE_PLANAR_3D_P(quantization_test_u8u8)

INST_TEST_CASE_BLOCKED_4D_P(quantization_test_f32f32)
INST_TEST_CASE_BLOCKED_4D_P(quantization_test_f32u8)
INST_TEST_CASE_BLOCKED_4D_P(quantization_test_f32s8)
INST_TEST_CASE_BLOCKED_4D_P(quantization_test_u8u8)

INST_TEST_CASE_PLANAR_4D_P(quantization_test_f32f32)
INST_TEST_CASE_PLANAR_4D_P(quantization_test_f32u8)
INST_TEST_CASE_PLANAR_4D_P(quantization_test_f32s8)
INST_TEST_CASE_PLANAR_4D_P(quantization_test_u8u8)

INST_TEST_CASE_PLANAR_5D_P(quantization_test_f32f32)
INST_TEST_CASE_PLANAR_5D_P(quantization_test_f32u8)
INST_TEST_CASE_PLANAR_5D_P(quantization_test_f32s8)
INST_TEST_CASE_PLANAR_5D_P(quantization_test_u8u8)

}
