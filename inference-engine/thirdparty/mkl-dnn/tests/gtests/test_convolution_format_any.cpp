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
#include "cpu_isa_traits.hpp"

#include "mkldnn.hpp"
namespace mkldnn {

using fmt = memory::format;

#define EXP_VALS_NUM 3
struct fmt_compare {
    fmt in;
    fmt exp[EXP_VALS_NUM];
};
struct conv_any_fmt_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    algorithm aalgorithm;
    fmt_compare src_fmt;
    fmt_compare weights_fmt;
    fmt_compare bias_fmt;
    fmt_compare dst_fmt;
    test_convolution_sizes_t test_cd;
};

template <typename data_t>
class convolution_any_fmt_test
        : public ::testing::TestWithParam<conv_any_fmt_test_params> {
protected:
    virtual bool FmtIsExp(const mkldnn_memory_format_t in, fmt *exp ) {
        for (int i = 0; i < EXP_VALS_NUM; i++)
            if (in == exp[i])
                return true;
        return false;
    }
    virtual void SetUp()
    {
        // Skip this test if the library cannot select blocked format a priori.
        // Currently blocking is supported only for sse42 and later CPUs.
        bool implementation_supports_blocking
            = impl::cpu::mayiuse(impl::cpu::sse42);
        if (!implementation_supports_blocking) return;

        conv_any_fmt_test_params p = ::testing::
                TestWithParam<conv_any_fmt_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aprop_kind, prop_kind::forward);
        ASSERT_EQ(p.aalgorithm, algorithm::convolution_direct);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        // Some format chekers
        ASSERT_NE(p.src_fmt.exp[0], fmt::any);
        ASSERT_NE(p.weights_fmt.exp[0], fmt::any);
        ASSERT_NE(p.bias_fmt.exp[0], fmt::any);
        ASSERT_NE(p.dst_fmt.exp[0], fmt::any);
        ASSERT_TRUE(
                p.src_fmt.in == fmt::any || p.src_fmt.in == p.src_fmt.exp[0]);
        ASSERT_TRUE(p.weights_fmt.in == fmt::any
                || p.weights_fmt.in == p.weights_fmt.exp[0]);
        ASSERT_TRUE(p.bias_fmt.in == fmt::any
                || p.bias_fmt.in == p.bias_fmt.exp[0]);
        ASSERT_TRUE(
                p.dst_fmt.in == fmt::any || p.dst_fmt.in == p.dst_fmt.exp[0]);

        test_convolution_sizes_t cd = p.test_cd;

        auto c_src_desc = create_md(
                { cd.mb, cd.ic, cd.ih, cd.iw }, data_type, p.src_fmt.in);
        auto c_weights_desc = cd.ng > 1 ?
                create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                        data_type, p.weights_fmt.in) :
                create_md({ cd.oc, cd.ic, cd.kh, cd.kw }, data_type,
                        p.weights_fmt.in);
        auto c_dst_desc = create_md(
                { cd.mb, cd.oc, cd.oh, cd.ow }, data_type, p.dst_fmt.in);

        bool with_bias = p.bias_fmt.in != fmt::format_undef;
        auto c_bias_desc = with_bias ?
                create_md({ cd.oc }, data_type, p.bias_fmt.in) :
                create_md({}, data_type, p.bias_fmt.in);

        auto conv_desc = with_bias ?
                convolution_forward::desc(p.aprop_kind, p.aalgorithm, c_src_desc,
                        c_weights_desc, c_bias_desc, c_dst_desc,
                        { cd.strh, cd.strw }, { cd.padh, cd.padw }, { cd.padh, cd.padw },
                        padding_kind::zero) :
                convolution_forward::desc(p.aprop_kind, p.aalgorithm, c_src_desc,
                        c_weights_desc, c_dst_desc, { cd.strh, cd.strw }, { cd.strh, cd.strw },
                        { cd.padh, cd.padw }, padding_kind::zero);

        auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

        ASSERT_TRUE(
                FmtIsExp(conv_prim_desc.src_primitive_desc().desc().data.format,
                        p.src_fmt.exp));
        ASSERT_TRUE(FmtIsExp(
                conv_prim_desc.weights_primitive_desc().desc().data.format,
                p.weights_fmt.exp));
        if (with_bias) {
            ASSERT_TRUE(FmtIsExp(
                    conv_prim_desc.bias_primitive_desc().desc().data.format,
                    p.bias_fmt.exp));
        }
        ASSERT_TRUE(
                FmtIsExp(conv_prim_desc.dst_primitive_desc().desc().data.format,
                        p.dst_fmt.exp));
    }
};

using conv_any_fmt_test_float = convolution_any_fmt_test<float>;
using conv_any_fmt_test_params_float = conv_any_fmt_test_params;

TEST_P(conv_any_fmt_test_float, TestsConvolutionAnyFmt)
{
}
#define ENGINE engine::kind::cpu
#define ALG algorithm::convolution_direct
#define PROP_KIND prop_kind::forward

#define ANY_X { fmt::any,    \
              { fmt::x, fmt::format_undef, fmt::format_undef } }
#define ANY_NCHW { fmt::any, \
              { fmt::nchw, fmt::format_undef, fmt::format_undef } }
#define ANY_OIHW { fmt::any, \
                 { fmt::oihw, fmt::format_undef, fmt::format_undef } }

#define ANY_OHWIxO { fmt::any,   \
                   { fmt::Ohwi8o, fmt::Ohwi16o, fmt::Oihw16o } }
#define ANY_NCHWxC { fmt::any,   \
                   { fmt::nChw8c, fmt::nChw16c, fmt::format_undef } }
#define ANY_OIHWxIxO { fmt::any, \
                     { fmt::OIhw8i8o, fmt::OIhw16i16o, fmt::format_undef } }
#define ANY_GOIHWxIxO { fmt::any,\
                      { fmt::gOIhw8i8o, fmt::gOIhw16i16o, fmt::format_undef } }

//INSTANTIATE_TEST_CASE_P(TestConvolutionAnyFmtForward, conv_any_fmt_test_float,
//    ::testing::Values(conv_any_fmt_test_params_float{ PROP_KIND, ENGINE, ALG,
//    ANY_NCHW, ANY_OIHW, ANY_X, ANY_NCHW,
//    { 2, 1, 4, 4, 4, 6, 4, 4, 3, 3, 1, 1, 1, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionAlexnetAnyFmtForwardxlocked, conv_any_fmt_test_float,
        ::testing::Values(
                conv_any_fmt_test_params_float{ PROP_KIND, ENGINE, ALG,
                        ANY_NCHW, ANY_OHWIxO, ANY_X, ANY_NCHWxC,
                        { 2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4 } },
                conv_any_fmt_test_params_float{ PROP_KIND, ENGINE, ALG,
                        ANY_NCHWxC, ANY_GOIHWxIxO, ANY_X, ANY_NCHWxC,
                        { 2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1 } },
                conv_any_fmt_test_params_float{ PROP_KIND, ENGINE, ALG,
                        ANY_NCHWxC, ANY_OIHWxIxO, ANY_X, ANY_NCHWxC,
                        { 2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_any_fmt_test_params_float{ PROP_KIND, ENGINE, ALG,
                        ANY_NCHWxC, ANY_GOIHWxIxO, ANY_X, ANY_NCHWxC,
                        { 2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_any_fmt_test_params_float{ PROP_KIND, ENGINE, ALG,
                    ANY_NCHWxC, ANY_GOIHWxIxO, ANY_X, ANY_NCHWxC,
                    { 2, 2, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1, 1 } }));
}
