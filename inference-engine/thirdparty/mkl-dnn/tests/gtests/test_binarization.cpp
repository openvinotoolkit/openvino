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

template <typename data_t>
struct binarization_test_params {
    engine::kind engine_kind;
    algorithm alg_kind;
    memory::format data_format;
    memory::dims dims;
};

template <typename src_data_t>
void check_binarization_fwd(const binarization_test_params<src_data_t> &p,
        const memory::desc &src_md, const memory &src, const memory &weights,
        const memory &output_low, const memory &output_high, const memory &dst) {
    auto src_data = (src_data_t*)src.get_data_handle();
    auto weights_data = (src_data_t*)weights.get_data_handle();
    auto output_low_data = (float*)output_low.get_data_handle();
    auto output_high_data = (float*)output_high.get_data_handle();
    auto dst_data = (uint8_t*)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc output_low_d = output_low.get_primitive_desc().desc();
    const memory::desc output_high_d = output_high.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    int N = src_md.data.ndims > 0 ? src_md.data.dims[0] : 1;
    int C = src_md.data.ndims > 1 ? src_md.data.dims[1] : 1;
    int H = src_md.data.ndims > 2 ? src_md.data.dims[2] : 1;
    int W = src_md.data.ndims > 3 ? src_md.data.dims[3] : 1;

    int nbits = 8;
    int CB = div_up(C, nbits);

    int padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];
    int padded_oc = dst_d.data.layout_desc.blocking.padding_dims[1];

    for (int n = 0; n < N; ++n) {
        for (int cb = 0; cb < CB; ++cb) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {

                    uint8_t bin_val = 0x00;
                    for (int c = cb * nbits, shift = 0; c < std::min(C, (cb + 1) * nbits); c++, shift++) {
                        int src_idx = n*padded_ic*H*W + c*H*W + h*W + w;
                        int wei_idx = c;

                        src_data_t s_val = src_data[map_index(src_d, src_idx)];
                        src_data_t w_val = weights_data[map_index(weights_d, wei_idx)];
                        src_data_t out_low = output_low_data[map_index(output_low_d, wei_idx)];
                        src_data_t out_high = output_high_data[map_index(output_high_d, wei_idx)];

                        auto bit = uint8_t((s_val > w_val) ? out_high : out_low);
                        bin_val |= (bit << shift);
                    }

                    int dst_idx = n*padded_oc*H*W + cb*nbits*H*W + h*W + w;
                    dst_idx = map_index(dst_d, dst_idx);

                    EXPECT_EQ(dst_data[dst_idx / nbits], bin_val);
                }
            }
        }
    }
}

template <typename src_data_t>
class binarization_test : public ::testing::TestWithParam<binarization_test_params<src_data_t>> {
private:

protected:
    virtual void SetUp() {
        auto p = ::testing::TestWithParam<binarization_test_params<src_data_t>>::GetParam();

        auto eng = engine(p.engine_kind, 0);
        auto src_data_type = data_traits<src_data_t>::data_type;

        memory::dims src_dims = memory::dims({p.dims[0], p.dims[1], p.dims[2], p.dims[3]});
        memory::dims wei_dims = memory::dims({src_dims[1]});
        memory::dims dst_dims = memory::dims({p.dims[0], p.dims[1], p.dims[2], p.dims[3]});

        auto src_desc = create_md(src_dims, src_data_type, p.data_format);
        auto weights_desc = create_md(wei_dims, src_data_type, memory::format::x);
        auto output_low_desc = create_md(wei_dims, src_data_type, memory::format::x);
        auto output_high_desc = create_md(wei_dims, src_data_type, memory::format::x);
        auto output_mask_desc = create_md(wei_dims, src_data_type, memory::format::x);
        auto dst_desc = create_md(dst_dims, memory::data_type::bin, p.data_format);

        auto src = test_memory(src_desc, eng);
        auto weights = test_memory(weights_desc, eng);
        auto output_low = test_memory(output_low_desc, eng);
        auto output_high = test_memory(output_high_desc, eng);
        auto output_mask = test_memory(output_mask_desc, eng);
        auto dst = test_memory(dst_desc, eng);

        fill_data<src_data_t>(src.get_size() / sizeof(src_data_t), (src_data_t *)src.get().get_data_handle(),
                              src_data_t(0), src_data_t(1));
        fill_data<src_data_t>(weights.get_size() / sizeof(src_data_t), (src_data_t *)weights.get().get_data_handle(),
                              src_data_t(0), src_data_t(1));
        fill_data<src_data_t>(output_low.get_size() / sizeof(src_data_t), (src_data_t *)output_low.get().get_data_handle(),
                              src_data_t(0), src_data_t(1));
        fill_data<uint8_t>(dst.get_size() / sizeof(uint8_t), (uint8_t*)dst.get().get_data_handle());

        src_data_t* p_output_low = (src_data_t *)output_low.get().get_data_handle();
        src_data_t* p_output_high = (src_data_t *)output_high.get().get_data_handle();
        uint32_t* p_output_mask = (uint32_t *)output_mask.get().get_data_handle();
        for (int i = 0; i < src_dims[1]; i++) {
            p_output_low[i] = p_output_low[i] >= 0 ? 1 : 0;
            p_output_high[i] = p_output_low[i] == 1 ? 0 : 1;
            p_output_mask[i] = p_output_high[i] == 1 ? 0xffffffff : 0x00000000;
        }

        std::vector<primitive> pipeline;
        auto binarization_desc = quantization_forward::desc(prop_kind::forward_training, p.alg_kind, 1, src_desc, weights_desc, output_high_desc, dst_desc);
        auto binarization_prim_desc = quantization_forward::primitive_desc(binarization_desc, eng);
        auto binarization = quantization_forward(binarization_prim_desc, src.get(), weights.get(), output_mask.get(), dst.get());

        pipeline.push_back(binarization);
        auto s = stream(stream::kind::lazy);
        s.submit(pipeline).wait();

        check_binarization_fwd(p, src_desc, src.get(), weights.get(), output_low.get(), output_high.get(), dst.get());
    }
};

using binarization_test_float = binarization_test<float>;
using binarization_test_params_float = binarization_test_params<float>;

TEST_P(binarization_test_float, TestsBinarization)
{
}

#define EXPAND(args) args

#define EXPAND_FORMATS(data) memory::format::data

#define ENGINE engine::kind::cpu

#define PARAMS(alg, data, mb, c, h, w) \
    binarization_test_params_float { ENGINE, algorithm::alg, \
    EXPAND_FORMATS(data), {mb, c, h, w} }

#define PARAMS_ALL_ALG(...) \
    EXPAND(PARAMS(binarization_depthwise, __VA_ARGS__))

#define INST_TEST_CASE(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, binarization_test_float, ::testing::Values(__VA_ARGS__))

INST_TEST_CASE(Simple_NHWC,
    PARAMS_ALL_ALG(nhwc, 2, 8, 4, 4),
    PARAMS_ALL_ALG(nhwc, 2, 16, 4, 4),
    PARAMS_ALL_ALG(nhwc, 2, 16, 8, 8),
    PARAMS_ALL_ALG(nhwc, 2, 16, 16, 8),
    PARAMS_ALL_ALG(nhwc, 2, 16, 10, 8),
    PARAMS_ALL_ALG(nhwc, 10, 10, 10, 10),
    PARAMS_ALL_ALG(nhwc, 256, 64, 8, 16),
    PARAMS_ALL_ALG(nhwc, 1, 1, 1, 1),
    PARAMS_ALL_ALG(nhwc, 3, 5, 7, 11),
    PARAMS_ALL_ALG(nhwc, 2, 129, 7, 4),
    PARAMS_ALL_ALG(nhwc, 2, 333, 8, 3)
);

}
