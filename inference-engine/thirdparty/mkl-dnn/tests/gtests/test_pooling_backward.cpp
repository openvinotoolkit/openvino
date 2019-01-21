/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"
namespace mkldnn {

struct test_pool_bwd_desc_t {
    int mb, c;
    int id, ih, iw;
    int od, oh, ow;
    int kd, kh, kw;
    int padf, padt, padl;
    int strd, strh, strw;
};

struct pool_bwd_test_params {
    engine::kind engine_kind;
    algorithm aalgorithm;
    memory::format diff_src_format;
    memory::format diff_dst_format;
    int ndims;
    test_pool_bwd_desc_t test_pd;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

template <typename data_t>
void check_pool_fwd(const pool_bwd_test_params &p, const memory &src,
        const memory &dst)
{
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    auto pd = p.test_pd;

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };
    size_t padded_c = src_d.data.layout_desc.blocking.padding_dims[1];

    mkldnn::impl::parallel_nd(pd.mb, pd.c, pd.od, pd.oh, pd.ow,
        [&](int n, int c, int od, int oh, int ow) {
            size_t oidx = (size_t)n * padded_c * pd.od * pd.oh * pd.ow
                    + (size_t)c * pd.od * pd.oh * pd.ow
                    + (size_t)od * pd.oh * pd.ow + (size_t)oh * pd.ow + ow;
            data_t out = dst_data[map_index(dst_d, oidx)];

            // match implementation for pooling_max: padding
            // is done with lowest value and not zero, it
            // affects the case when kernel slips into
            // the padding area entirely
            data_t out_ref = (p.aalgorithm == pooling_max) ?
                    std::numeric_limits<data_t>::lowest() :
                    data_t(0);
            bool is_initialized = false;

            auto id_start = apply_offset(od*pd.strd, pd.padf);
            auto ih_start = apply_offset(oh*pd.strh, pd.padt);
            auto iw_start = apply_offset(ow*pd.strw, pd.padl);
            auto id_end = std::min(od*pd.strd - pd.padf + pd.kd, pd.id);
            auto ih_end = std::min(oh*pd.strh - pd.padt + pd.kh, pd.ih);
            auto iw_end = std::min(ow*pd.strw - pd.padl + pd.kw, pd.iw);

            auto num_summands = p.aalgorithm != pooling_avg_exclude_padding
                ? pd.kw*pd.kh*pd.kd
                : (ih_end - ih_start) * (iw_end - iw_start)
                    * (id_end - id_start);

            for (int id = id_start; id < id_end; ++id)
            for (int ih = ih_start; ih < ih_end; ++ih)
            for (int iw = iw_start; iw < iw_end; ++iw) {
                size_t iidx = (size_t)n * padded_c * pd.id * pd.ih * pd.iw
                        + (size_t)c * pd.id * pd.ih * pd.iw
                        + (size_t)id * pd.ih * pd.iw
                        + (size_t)ih * pd.iw + iw;

                data_t d = src_data[map_index(src_d, iidx)];
                if (p.aalgorithm == pooling_max) {
                    if (!is_initialized) {
                        out_ref = d;
                        is_initialized = true;
                    } else {
                        if (out_ref < d) out_ref = d;
                    }
                } else if (p.aalgorithm == pooling_avg_include_padding
                    || p.aalgorithm == pooling_avg_exclude_padding) {
                    out_ref += d;
                }
            }

            if (p.aalgorithm == pooling_avg_include_padding ||
                p.aalgorithm == pooling_avg_exclude_padding) {
                out_ref /= num_summands;
            }
            EXPECT_NEAR(out, out_ref, 1e-6f);
        }
    );
}

template <typename data_t>
void check_pool_bwd(const pool_bwd_test_params &p, const memory &diff_src,
        const memory &diff_dst, const memory &ws)
{
    data_t *diff_src_data = (data_t *)diff_src.get_data_handle();
    data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();

    auto ws_data = [=](size_t idx) -> int {
        auto w = (unsigned char *)ws.get_data_handle();
        if (w == nullptr) return -1;
        if (ws.get_primitive_desc().desc().data.data_type == mkldnn_u8)
            return (int)w[idx];
        else
            return ((int *)w)[idx];
    };

    const memory::desc diff_src_d = diff_src.get_primitive_desc().desc();
    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();
    const memory::desc ws_d = ws.get_primitive_desc().desc();

    auto pd = p.test_pd;
    std::vector<data_t>
        ref_diff_src_vec((size_t)pd.mb * pd.c * pd.id * pd.ih * pd.iw);
    data_t *ref_diff_src = &ref_diff_src_vec[0];

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    mkldnn::impl::parallel_nd((size_t)pd.mb * pd.c * pd.id * pd.ih * pd.iw,
        [&](size_t i) { ref_diff_src[i] = 0.; }
    );

    mkldnn::impl::parallel_nd(pd.mb, pd.c, [&](int n, int c) {
        for (int od = 0; od < pd.od; od++)
        for (int oh = 0; oh < pd.oh; oh++)
        for (int ow = 0; ow < pd.ow; ow++) {
            size_t oidx = (size_t)n * pd.c * pd.od * pd.oh * pd.ow
                    + (size_t)c * pd.od * pd.oh * pd.ow
                    + (size_t)od * pd.oh * pd.ow + (size_t)oh * pd.ow + ow;
            data_t diff_dst = diff_dst_data[map_index(diff_dst_d, oidx)];
            if (p.aalgorithm == pooling_max) {
                int kw_max = ws_data(map_index(ws_d, oidx)) % pd.kw;
                int kh_max = (ws_data(map_index(ws_d, oidx)) / pd.kw) % pd.kh;
                int kd_max = (ws_data(map_index(ws_d, oidx)) / pd.kw) / pd.kh;
                for (int kd = 0; kd < pd.kd; kd++)
                for (int kh = 0; kh < pd.kh; kh++)
                for (int kw = 0; kw < pd.kw; kw++) {
                    int iw = ow * pd.strw - pd.padl + kw;
                    int ih = oh * pd.strh - pd.padt + kh;
                    int id = od * pd.strd - pd.padf + kd;
                    if (iw < 0 || iw >= pd.iw) continue;
                    if (ih < 0 || ih >= pd.ih) continue;
                    if (id < 0 || id >= pd.id) continue;
                    size_t iidx = (size_t)n * pd.c * pd.id * pd.ih * pd.iw
                            + (size_t)c * pd.id * pd.ih * pd.iw
                            + (size_t)id * pd.ih * pd.iw
                            + (size_t)ih * pd.iw + iw;

                    if (kh == kh_max && kw == kw_max && kd == kd_max)
                        ref_diff_src[iidx] += diff_dst;
                }
            } else if (p.aalgorithm == pooling_avg_include_padding
                || p.aalgorithm == pooling_avg_exclude_padding) {
                auto id_start = apply_offset(od*pd.strd, pd.padf);
                auto ih_start = apply_offset(oh*pd.strh, pd.padt);
                auto iw_start = apply_offset(ow*pd.strw, pd.padl);
                auto id_end =
                    std::min(od*pd.strd - pd.padf + pd.kd, pd.id);
                auto ih_end =
                    std::min(oh*pd.strh - pd.padt + pd.kh, pd.ih);
                auto iw_end =
                    std::min(ow*pd.strw - pd.padl + pd.kw, pd.iw);

                auto num_summands = (p.aalgorithm != pooling_avg_exclude_padding)
                    ? pd.kw*pd.kh*pd.kd
                    : (ih_end - ih_start) * (iw_end - iw_start)
                        * (id_end - id_start);

                for (int id = id_start; id < id_end; id++) {
                    for (int ih = ih_start; ih < ih_end; ih++) {
                        for (int iw = iw_start; iw < iw_end; iw++) {
                            size_t iidx = (size_t)n * pd.c * pd.id * pd.ih
                                            * pd.iw
                                    + (size_t)c * pd.id * pd.ih * pd.iw
                                    + (size_t)id * pd.ih * pd.iw
                                    + (size_t)ih * pd.iw + iw;
                            ref_diff_src[iidx] += diff_dst / num_summands;
                        }
                    }
                }
            }
        }
    });

    mkldnn::impl::parallel_nd((size_t)pd.mb * pd.c * pd.id * pd.ih * pd.iw,
        [&](size_t i) {
            EXPECT_NEAR(ref_diff_src[i],
                    diff_src_data[map_index(diff_src_d, i)], 1e-5f);
        }
    );
}

template <typename data_t>
class pooling_bwd_test : public ::testing::TestWithParam<pool_bwd_test_params> {
private:
    std::shared_ptr<memory::desc> src_desc;
    std::shared_ptr<memory::desc> dst_desc;
    std::shared_ptr<memory> workspace;
    std::shared_ptr<pooling_forward::primitive_desc> pool_prim_desc;
    pool_bwd_test_params p;
    memory::dims padR_3d;
    memory::dims padR_2d;
    std::shared_ptr<engine> eng;
    memory::data_type data_type;

protected:
    virtual void SetUp() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        test_pool_bwd_desc_t pd = p.test_pd;

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        eng.reset(new engine(p.engine_kind, 0));
        data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        if (p.ndims == 5)
        {
            src_desc.reset(new memory::desc(
                { pd.mb, pd.c, pd.id, pd.ih, pd.iw }, data_type,
                p.diff_src_format));
            dst_desc.reset(new memory::desc(
                { pd.mb, pd.c, pd.od, pd.oh, pd.ow }, data_type,
                p.diff_dst_format));
        } else {
            src_desc.reset(new memory::desc(
                { pd.mb, pd.c, pd.ih, pd.iw }, data_type, p.diff_src_format));
            dst_desc.reset(new memory::desc(
                { pd.mb, pd.c, pd.oh, pd.ow }, data_type, p.diff_dst_format));
        }

        // calculate right padding exactly
        padR_2d = {
            right_padding(pd.ih, pd.oh, pd.kh, pd.padt, pd.strh),
            right_padding(pd.iw, pd.ow, pd.kw, pd.padl, pd.strw)
        };
        padR_3d = {
            right_padding(pd.id, pd.od, pd.kd, pd.padf, pd.strd),
            right_padding(pd.ih, pd.oh, pd.kh, pd.padt, pd.strh),
            right_padding(pd.iw, pd.ow, pd.kw, pd.padl, pd.strw)
        };

        Forward();
        Backward();
    }

    void Forward()
    {
        std::shared_ptr<memory> src;
        std::shared_ptr<memory> dst;

        test_pool_bwd_desc_t pd = p.test_pd;

        auto pool_desc = (p.ndims == 5)
            ? pooling_forward::desc(prop_kind::forward_training,
                    p.aalgorithm, *src_desc, *dst_desc,
                    {pd.strd, pd.strh, pd.strw},
                    {pd.kd, pd.kh, pd.kw}, {pd.padf, pd.padt, pd.padl},
                    padR_3d, padding_kind::zero)
            : pooling_forward::desc(prop_kind::forward_training,
                    p.aalgorithm, *src_desc, *dst_desc, {pd.strh, pd.strw},
                    {pd.kh, pd.kw}, {pd.padt, pd.padl}, padR_2d,
                    padding_kind::zero);

        pool_prim_desc.reset(
                new pooling_forward::primitive_desc(pool_desc, *eng));

        bool with_workspace = p.aalgorithm == pooling_max;
        auto p_workspace_desc = with_workspace
            ? pool_prim_desc->workspace_primitive_desc()
            : memory::primitive_desc( {{}, data_type, p.diff_dst_format}, *eng);

        src.reset(new memory({*src_desc, *eng}));
        workspace.reset(new  memory(p_workspace_desc));
        dst.reset(new memory({*dst_desc, *eng}));

        fill_data<data_t>(
                src->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)src->get_data_handle());
        fill_data<data_t>(
                dst->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)dst->get_data_handle());
        check_zero_tail<data_t>(1, *src);
        check_zero_tail<data_t>(1, *dst);

        auto pool = with_workspace
            ? pooling_forward(*pool_prim_desc, *src, *dst, *workspace)
            : pooling_forward(*pool_prim_desc, *src, *dst);

        std::vector<primitive> pipeline;
        pipeline.push_back(pool);

        stream(stream::kind::lazy).submit(pipeline).wait();
        check_zero_tail<data_t>(0, *dst);
        check_pool_fwd<data_t>(p, *src, *dst);
    }

    void Backward()
    {
        std::shared_ptr<memory> diff_src;
        std::shared_ptr<memory> diff_dst;

        test_pool_bwd_desc_t pd = p.test_pd;

        auto pool_bwd_desc = (p.ndims == 5)
            ? pooling_backward::desc(p.aalgorithm, *src_desc, *dst_desc,
                    {pd.strd, pd.strh, pd.strw}, {pd.kd, pd.kh, pd.kw},
                    {pd.padf, pd.padt, pd.padl}, padR_3d, padding_kind::zero)
        : pooling_backward::desc(p.aalgorithm, *src_desc, *dst_desc,
                {pd.strh, pd.strw}, {pd.kh, pd.kw}, {pd.padt, pd.padl},
                padR_2d, padding_kind::zero);

        auto pool_bwd_prim_desc = pooling_backward::primitive_desc(
                pool_bwd_desc, *eng, *pool_prim_desc);

        bool with_workspace = p.aalgorithm == pooling_max;

        diff_src.reset(new memory({*src_desc, *eng}));
        diff_dst.reset(new memory({*dst_desc, *eng}));

        fill_data<data_t>(
                diff_dst->get_primitive_desc().get_size()/ sizeof(data_t),
                (data_t *)diff_dst->get_data_handle());
        fill_data<data_t>(
                diff_src->get_primitive_desc().get_size()/ sizeof(data_t),
                (data_t *)diff_src->get_data_handle());
        check_zero_tail<data_t>(1, *diff_dst);
        check_zero_tail<data_t>(1, *diff_src);
        auto pool_bwd = with_workspace
            ? pooling_backward(pool_bwd_prim_desc, *diff_dst, *workspace,
                    *diff_src)
            : pooling_backward(pool_bwd_prim_desc, *diff_dst, *diff_src);

        std::vector<primitive> pipeline2 = {pool_bwd};

        stream(stream::kind::lazy).submit(pipeline2).wait();
        check_zero_tail<data_t>(0, *diff_src);
        check_pool_bwd<data_t>(p, *diff_src, *diff_dst, *workspace);
    }
};

using pooling_bwd_test_float = pooling_bwd_test<float>;
using pool_bwd_test_params_float = pool_bwd_test_params;

#define EXPAND_SIZES_3D(...) 5, { __VA_ARGS__ }
#define EXPAND_SIZES_2D(mb,ic,ih,iw,oh,ow,kh,kw,padt,padl,strh,strw) \
    4, { mb,ic,1,ih,iw,1,oh,ow,1,kh,kw,0,padt,padl,1,strh,strw }

TEST_P(pooling_bwd_test_float, TestsPoolingBackward)
{
}

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardZeroDim, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, 0, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 )},
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 0, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 )},
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, 4, 0, 4, 4, 4, 3, 3, 1, 1, 1, 1 )}
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardEF, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, -4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 ),
            true, mkldnn_invalid_arguments},
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( -2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 ),
            true, mkldnn_invalid_arguments},
            pool_bwd_test_params_float{ engine::kind::cpu,
            eltwise_square, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 ),
            true, mkldnn_invalid_arguments}
            ));

INSTANTIATE_TEST_CASE_P(
        TestPooling_nChw16c_padded, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D(4, 17,  6,  6,  7,  7, 2, 2, 1, 1, 1, 1) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D(4, 23, 60, 60, 31, 31, 3, 4, 1, 1, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D(4, 14, 60, 60, 31, 31, 3, 2, 1, 1, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D(4, 17, 60, 60, 31, 31, 4, 3, 1, 1, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D(4, 14, 60, 60, 31, 31, 2, 3, 1, 1, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D(4, 28, 60, 60, 31, 31, 4, 2, 1, 1, 2, 2) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPooling_nChw8c_padded, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(4, 5,  6,  6,  7,  7, 2, 2, 1, 1, 1, 1) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(4, 23, 60, 60, 31, 31, 3, 4, 1, 1, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(4, 14, 60, 60, 31, 31, 3, 2, 1, 1, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(4, 17, 60, 60, 31, 31, 4, 3, 1, 1, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(4, 14, 60, 60, 31, 31, 2, 3, 1, 1, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(4, 28, 60, 60, 31, 31, 4, 2, 1, 1, 2, 2) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardMaxKernelSlipsToPadding, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 1, 16, 10, 10, 6, 6, 5, 5, 10, 10, 5, 5 ) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nhwc,
            memory::format::nhwc, EXPAND_SIZES_2D( 1, 16, 10, 10, 6, 6, 5, 5, 10, 10, 5, 5 ) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 1, 16, 10, 10, 6, 6, 5, 5, 10, 10, 5, 5 ) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 1, 16, 10, 10, 6, 6, 5, 5, 10, 10, 5, 5 ) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPooling3D_nCdhw16c, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nCdhw16c,
            memory::format::nCdhw16c, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30, 2, 3, 4, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nCdhw16c,
            memory::format::nCdhw16c, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nCdhw16c,
            memory::format::nCdhw16c, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nCdhw16c,
            memory::format::nCdhw16c, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nCdhw16c,
            memory::format::nCdhw16c, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nCdhw16c,
            memory::format::nCdhw16c, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPooling3D_ncdhw, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::ncdhw,
            memory::format::ncdhw, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30, 2, 3, 4, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::ncdhw,
            memory::format::ncdhw, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::ncdhw,
            memory::format::ncdhw, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::ncdhw,
            memory::format::ncdhw, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::ncdhw,
            memory::format::ncdhw, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::ncdhw,
            memory::format::ncdhw, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPooling3D_ndhwc, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::ndhwc,
            memory::format::ndhwc, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30, 2, 3, 4, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::ndhwc,
            memory::format::ndhwc, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::ndhwc,
            memory::format::ndhwc, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::ndhwc,
            memory::format::ndhwc, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::ndhwc,
            memory::format::ndhwc, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::ndhwc,
            memory::format::ndhwc, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPooling3D_nCdhw8c, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nCdhw8c,
            memory::format::nCdhw8c, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30, 2, 3, 4, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nCdhw8c,
            memory::format::nCdhw8c, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nCdhw8c,
            memory::format::nCdhw8c, EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3, 1, 1, 1, 2, 2, 2) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nCdhw8c,
            memory::format::nCdhw8c, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nCdhw8c,
            memory::format::nCdhw8c, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) },
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nCdhw8c,
            memory::format::nCdhw8c, EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3, 1, 1, 1, 1, 1, 1) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardMax3DunetNCDHW, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::ncdhw,
            memory::format::ncdhw, EXPAND_SIZES_3D(1, 64,  64, 64, 64, 64, 64, 64, 2, 2, 2, 0, 0, 0, 1, 1, 1) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::ncdhw,
            memory::format::ncdhw, EXPAND_SIZES_3D(1, 128, 28, 28, 28, 28, 28, 28, 2, 2, 2, 0, 0, 0, 1, 1, 1) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::ncdhw,
            memory::format::ncdhw, EXPAND_SIZES_3D(1, 256, 12, 12, 12, 12, 12, 12, 2, 2, 2, 0, 0, 0, 1, 1, 1) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardMax3DunetNDHWC, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::ndhwc,
            memory::format::ndhwc, EXPAND_SIZES_3D(1, 64,  64, 64, 64, 64, 64, 64, 2, 2, 2, 0, 0, 0, 1, 1, 1) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::ndhwc,
            memory::format::ndhwc, EXPAND_SIZES_3D(1, 128, 28, 28, 28, 28, 28, 28, 2, 2, 2, 0, 0, 0, 1, 1, 1) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::ndhwc,
            memory::format::ndhwc, EXPAND_SIZES_3D(1, 256, 12, 12, 12, 12, 12, 12, 2, 2, 2, 0, 0, 0, 1, 1, 1) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardMaxAlexNetNCHW, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 ) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 ) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 ) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardMaxCIFAR10NCHW, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 ) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 ) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 ) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardMax, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1 ) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, 2, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1 ) },
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nchw,
            memory::format::nchw, EXPAND_SIZES_2D( 2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 ) }
            ));


INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardMaxBlocked, pooling_bwd_test_float, ::testing::Values(

            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 1, 8, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 ) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardAvgBlocked, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 5, 5, 2, 2, 3, 3, 0, 0, 2, 2 ) }
            , pool_bwd_test_params_float{engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 32, 5, 5, 2, 2, 3, 3, 0, 0, 2, 2 ) }
            , pool_bwd_test_params_float{engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 8, 3, 2, 2, 2, 3, 3, 1, 1, 2, 1 ) }
            , pool_bwd_test_params_float{engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 2, 8, 3, 2, 2, 2, 3, 3, 1, 1, 2, 1 ) }

            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardMaxBlocked16, pooling_bwd_test_float, ::testing::Values(

            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 1, 16, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 ) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardAvgBlocked16, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1 ) }
            , pool_bwd_test_params_float{engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 5, 5, 2, 2, 3, 3, 0, 0, 2, 2 ) }
            , pool_bwd_test_params_float{engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 32, 5, 5, 2, 2, 3, 3, 0, 0, 2, 2 ) }
            , pool_bwd_test_params_float{engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 16, 3, 2, 2, 2, 3, 3, 1, 1, 2, 1 ) }
            , pool_bwd_test_params_float{engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 2, 16, 3, 2, 2, 2, 3, 3, 1, 1, 2, 1 ) }

            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardMaxBlockedPerf, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 ) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardAvgBlockedPerf, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D( 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 ) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardMaxBlocked16Perf, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_max, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 ) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardAvgBlocked16Perf, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_include_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 ) }
            , pool_bwd_test_params_float{ engine::kind::cpu,
            pooling_avg_exclude_padding, memory::format::nChw16c,
            memory::format::nChw16c, EXPAND_SIZES_2D( 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 ) }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBackwardAsymmPadding, pooling_bwd_test_float, ::testing::Values(
            pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1) }

            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2) }

            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2) }

            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2) }

            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2) }

            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_max, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2) }
            ,pool_bwd_test_params_float{
            engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
            memory::format::nChw8c, EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2) }

            ));

}
