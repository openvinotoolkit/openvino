/*******************************************************************************
* Copyright 2018 Intel Corporation
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

template <typename T> inline T scale_shift_fwd(T s_val, T w_val, T b_val) {
    return s_val*w_val + b_val;
}

template <typename T> inline T prelu_fwd(T s_val, T w_val) {
    return s_val >= 0 ? s_val : w_val*s_val;
}

template <typename data_t>
struct depthwise_test_params {
    engine::kind engine_kind;
    algorithm alg_kind;
    memory::format data_format;
    memory::dims dims;
};

template <typename data_t>
void check_depthwise_fwd(const depthwise_test_params<data_t> &p,
        const memory::desc &md, const memory &src, const memory &weights,
        const memory &bias, bool with_bias, const memory &dst)
{
    data_t *src_data     = (data_t *)src.get_data_handle();
    data_t *weights_data = (data_t *)weights.get_data_handle();
    data_t *bias_data    = with_bias ? (data_t *)bias.get_data_handle() : nullptr;
    data_t *dst_data     = (data_t *)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    ASSERT_EQ(md.data.data_type, memory::data_type::f32); // TODO: type assert

    int N = md.data.ndims > 0 ? md.data.dims[0] : 1;
    int C = md.data.ndims > 1 ? md.data.dims[1] : 1;
    int H = md.data.ndims > 2 ? md.data.dims[2] : 1;
    int W = md.data.ndims > 3 ? md.data.dims[3] : 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = n*C*H*W + c*H*W + h*W + w;

                    data_t s_val = src_data[map_index(src_d, idx)];
                    data_t w_val = weights_data[map_index(weights_d, c)];
                    data_t b_val = with_bias ? bias_data[map_index(bias.get_primitive_desc().desc(), c)] : 0;

                    data_t ref_d = 0;
                    switch (p.alg_kind) {
                        case depthwise_scale_shift:
                            ref_d = scale_shift_fwd(s_val, w_val, b_val);
                            break;
                        case depthwise_prelu:
                            ref_d = prelu_fwd(s_val, w_val);
                            break;
                        default:
                            assert(!"unknown alg_kind");
                    }

                    EXPECT_NEAR(dst_data[map_index(dst_d, idx)], ref_d, 1.e-6);
                }
            }
        }
    }
}

template <typename data_t>
class depthwise_test : public ::testing::TestWithParam<depthwise_test_params<data_t>> {
private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> bias;
    std::shared_ptr<memory> dst;
    std::shared_ptr<memory> workspace;
    std::shared_ptr<memory::desc> data_desc;
    std::shared_ptr<memory::desc> weights_desc;
    std::shared_ptr<memory::desc> bias_desc;
    std::shared_ptr<depthwise_forward::primitive_desc> depthwise_prim_desc;
    depthwise_test_params<data_t> p;
    std::shared_ptr<engine> eng;
    memory::data_type data_type;
    int data_size;
    int weights_size;

protected:
    virtual void SetUp() {
        p = ::testing::TestWithParam<depthwise_test_params<data_t>>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        eng.reset(new engine(p.engine_kind, 0));

        data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        data_size = p.dims[0] * p.dims[1] * p.dims[2] * p.dims[3];
        weights_size = p.dims[1];

        Forward();
    }

    void Forward() {
        bool with_bias = p.alg_kind == depthwise_scale_shift;

        memory::dims dims = p.data_format == mkldnn_nc ? memory::dims({p.dims[0], p.dims[1]}) : p.dims;

        data_desc.reset(new memory::desc(dims, data_type, p.data_format));
        src.reset(new memory({*data_desc, *eng}));
        dst.reset(new memory({*data_desc, *eng}));
        fill_data<data_t>(data_size, (data_t *)src->get_data_handle(),
                          data_t(0), data_t(1));

        weights_desc.reset(new memory::desc({dims[1]}, data_type, memory::format::x));
        weights.reset(new memory({*weights_desc, *eng}));
        fill_data<data_t>(weights_size, (data_t *)weights->get_data_handle(),
                          data_t(0), data_t(1));

        if (with_bias) {
            bias_desc.reset(new memory::desc({dims[1]}, data_type, memory::format::x));
            bias.reset(new memory({*bias_desc, *eng}));
            fill_data<data_t>(weights_size, (data_t *) bias->get_data_handle(),
                              data_t(0), data_t(1));
        }

        std::vector<primitive> pipeline;
        auto depthwise_desc = with_bias
                              ? depthwise_forward::desc(prop_kind::forward_training, p.alg_kind, *data_desc, *data_desc, *weights_desc, *bias_desc)
                              : depthwise_forward::desc(prop_kind::forward_training, p.alg_kind, *data_desc, *data_desc, *weights_desc);
        depthwise_prim_desc.reset(new depthwise_forward::primitive_desc(depthwise_desc, *eng));

        auto depthwise = with_bias
                         ? depthwise_forward(*depthwise_prim_desc, *src, *weights, *bias, *dst)
                         : depthwise_forward(*depthwise_prim_desc, *src, *weights, *dst);

        pipeline.push_back(depthwise);
        auto s = stream(stream::kind::lazy);
        s.submit(pipeline).wait();

        check_depthwise_fwd(p, *data_desc, *src, *weights, *bias, with_bias, *dst);
    }
};

using depthwise_test_float = depthwise_test<float>;
using depthwise_test_params_float = depthwise_test_params<float>;

TEST_P(depthwise_test_float, TestsDepthwise)
{
}

#define EXPAND(args) args

#define EXPAND_FORMATS(data) memory::format::data

#define ENGINE engine::kind::cpu

#define PARAMS(alg, data, mb, c, h, w) \
    depthwise_test_params_float { ENGINE, algorithm::alg, \
    EXPAND_FORMATS(data), {mb, c, h, w} }

#define PARAMS_ALL_ALG(...) \
    EXPAND(PARAMS(depthwise_scale_shift, __VA_ARGS__)), \
    EXPAND(PARAMS(depthwise_prelu, __VA_ARGS__))

#define INST_TEST_CASE(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, depthwise_test_float, ::testing::Values(__VA_ARGS__))

INST_TEST_CASE(Simple_NC,
    PARAMS_ALL_ALG(nc, 2, 8, 1, 1),
    PARAMS_ALL_ALG(nc, 2, 16, 1, 1),
    PARAMS_ALL_ALG(nc, 2, 16, 1, 1),
    PARAMS_ALL_ALG(nc, 2, 16, 1, 1),
    PARAMS_ALL_ALG(nc, 2, 16, 1, 1),
    PARAMS_ALL_ALG(nc, 10, 10, 1, 1),
    PARAMS_ALL_ALG(nc, 256, 64, 1, 1),
    PARAMS_ALL_ALG(nc, 1, 1, 1, 1),
    PARAMS_ALL_ALG(nc, 3, 5, 1, 1)
);

INST_TEST_CASE(Simple_NCHW,
    PARAMS_ALL_ALG(nchw, 2, 8, 4, 4),
    PARAMS_ALL_ALG(nchw, 2, 16, 4, 4),
    PARAMS_ALL_ALG(nchw, 2, 16, 8, 8),
    PARAMS_ALL_ALG(nchw, 2, 16, 16, 8),
    PARAMS_ALL_ALG(nchw, 2, 16, 10, 8),
    PARAMS_ALL_ALG(nchw, 10, 10, 10, 10),
    PARAMS_ALL_ALG(nchw, 256, 64, 8, 16),
    PARAMS_ALL_ALG(nchw, 1, 1, 1, 1),
    PARAMS_ALL_ALG(nchw, 3, 5, 7, 11)
);

INST_TEST_CASE(Simple_Blocked,
    PARAMS_ALL_ALG(nChw8c, 2, 8, 4, 4),
    PARAMS_ALL_ALG(nChw8c, 2, 16, 4, 4),
    PARAMS_ALL_ALG(nChw8c, 2, 16, 8, 8),
    PARAMS_ALL_ALG(nChw8c, 2, 16, 16, 8),
    PARAMS_ALL_ALG(nChw8c, 2, 32, 10, 8),
    PARAMS_ALL_ALG(nChw8c, 256, 64, 8, 16)
);

INST_TEST_CASE(Simple_Blocked16,
    PARAMS_ALL_ALG(nChw16c, 2, 16, 4, 4),
    PARAMS_ALL_ALG(nChw16c, 2, 16, 8, 8),
    PARAMS_ALL_ALG(nChw16c, 2, 16, 16, 8),
    PARAMS_ALL_ALG(nChw16c, 2, 32, 10, 8),
    PARAMS_ALL_ALG(nChw16c, 256, 64, 8, 16)
);

}
