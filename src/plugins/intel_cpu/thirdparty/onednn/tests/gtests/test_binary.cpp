/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using tag = memory::format_tag;
using data_type = memory::data_type;

struct binary_test_params_t {
    std::vector<tag> srcs_format;
    tag dst_format;
    algorithm aalgorithm;
    memory::dims dims;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename src0_data_t, typename src1_data_t = src0_data_t,
        typename dst_data_t = src0_data_t>
class binary_test_t : public ::testing::TestWithParam<binary_test_params_t> {
private:
    binary_test_params_t p;
    data_type src0_dt, src1_dt, dst_dt;

protected:
    void SetUp() override {
        src0_dt = data_traits<src0_data_t>::data_type;
        src1_dt = data_traits<src1_data_t>::data_type;
        dst_dt = data_traits<dst_data_t>::data_type;

        p = ::testing::TestWithParam<binary_test_params_t>::GetParam();

        SKIP_IF(unsupported_data_type(src0_dt),
                "Engine does not support this data type.");

        SKIP_IF(unsupported_data_type(src1_dt),
                "Engine does not support this data type.");

        SKIP_IF(unsupported_data_type(dst_dt),
                "Engine does not support this data type.");

        SKIP_IF_CUDA(
                !cuda_check_data_types_combination(src0_dt, src1_dt, dst_dt),
                "Engine does not support this data type combination.");

        for (auto tag : p.srcs_format) {
            MAYBE_UNUSED(tag);
            SKIP_IF_CUDA(!cuda_check_format_tag(tag),
                    "Unsupported source format tag");
        }
        SKIP_IF_CUDA(!cuda_check_format_tag(p.dst_format),
                "Unsupported destination format tag");

        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    bool cuda_check_data_types_combination(
            data_type src0_dt, data_type src1_dt, data_type dst_dt) {
        bool correct_input_dt = src0_dt == data_type::f32 || src0_dt == dst_dt
                || dst_dt == data_type::f32;
        bool inputs_same_dt = src0_dt == src1_dt;

        return inputs_same_dt && correct_input_dt;
    }

    bool cuda_check_format_tag(tag atag) {
        return atag == tag::abcd || atag == tag::acdb;
    }

    void Test() {
        auto eng = get_test_engine();
        auto strm = make_stream(eng);

        // binary specific types and values
        using op_desc_t = binary::desc;
        using pd_t = binary::primitive_desc;
        allows_attr_t aa {false};
        aa.scales = true;
        aa.po_sum = !is_nvidia_gpu(eng);
        aa.po_eltwise = !is_nvidia_gpu(eng);
        aa.po_binary = !is_nvidia_gpu(eng);
        std::vector<memory::desc> srcs_md;
        std::vector<memory> srcs;

        for (int i_case = 0;; ++i_case) {
            memory::dims dims_B = p.dims;
            if (i_case == 0) {
            } else if (i_case == 1) {
                dims_B[0] = 1;
            } else if (i_case == 2) {
                dims_B[1] = 1;
                dims_B[2] = 1;
            } else if (i_case == 3) {
                dims_B[0] = 1;
                dims_B[2] = 1;
                dims_B[3] = 1;
            } else if (i_case == 4) {
                dims_B[0] = 1;
                dims_B[1] = 1;
                dims_B[2] = 1;
                dims_B[3] = 1;
            } else {
                break;
            }

            auto desc_A = memory::desc(p.dims, src0_dt, p.srcs_format[0]);
            // TODO: try to fit "reshape" logic here.
            auto desc_B = memory::desc(dims_B, src1_dt, memory::dims());
            auto desc_C = memory::desc(p.dims, dst_dt, p.dst_format);

            const dnnl::impl::memory_desc_wrapper mdw_desc_A(desc_A.data);
            const bool has_zero_dim = mdw_desc_A.has_zero_dim();

            // default op desc ctor
            auto op_desc = op_desc_t();
            // regular op desc ctor
            op_desc = op_desc_t(p.aalgorithm, desc_A, desc_B, desc_C);

            // default pd ctor
            auto pd = pd_t();
            // regular pd ctor
            ASSERT_NO_THROW(pd = pd_t(op_desc, eng));
            // test all pd ctors
            if (!has_zero_dim)
                test_fwd_pd_constructors<op_desc_t, pd_t>(op_desc, pd, aa);

            // default primitive ctor
            auto prim = binary();
            // regular primitive ctor
            prim = binary(pd);

            // query for descs from pd
            const auto src0_desc = pd.src_desc(0);
            const auto src1_desc = pd.src_desc(1);
            const auto dst_desc = pd.dst_desc();
            const auto workspace_desc = pd.workspace_desc();

            ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC_0)
                    == src0_desc);
            ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC_1)
                    == src1_desc);
            ASSERT_TRUE(
                    pd.query_md(query::exec_arg_md, DNNL_ARG_DST) == dst_desc);

            // check primitive returns zero_md for all rest md
            ASSERT_TRUE(pd.weights_desc().is_zero());
            ASSERT_TRUE(pd.diff_src_desc().is_zero());
            ASSERT_TRUE(pd.diff_dst_desc().is_zero());
            ASSERT_TRUE(pd.diff_weights_desc().is_zero());

            const auto test_engine = pd.get_engine();

            auto mem_A = test::make_memory(src0_desc, test_engine);
            auto mem_B = test::make_memory(src1_desc, test_engine);
            auto mem_C = test::make_memory(dst_desc, test_engine);
            auto mem_ws = test::make_memory(workspace_desc, test_engine);

            fill_data<src0_data_t>(
                    src0_desc.get_size() / sizeof(src0_data_t), mem_A);
            fill_data<src1_data_t>(
                    src1_desc.get_size() / sizeof(src1_data_t), mem_B);
            // Remove zeroes in src1 to avoid division by zero
            remove_zeroes<src1_data_t>(mem_B);

            prim.execute(strm,
                    {{DNNL_ARG_SRC_0, mem_A}, {DNNL_ARG_SRC_1, mem_B},
                            {DNNL_ARG_DST, mem_C},
                            {DNNL_ARG_WORKSPACE, mem_ws}});
            strm.wait();
        }
    }
};

struct binary_attr_test_t
    : public ::testing::TestWithParam<
              std::tuple<memory::dims, memory::dims, memory::format_tag>> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(
        binary_attr_test_t, TestBinaryShouldCallSameImplementationWithPostops) {
    auto engine_kind = get_test_engine_kind();
    SKIP_IF(!DNNL_X64 || engine_kind != engine::kind::cpu,
            "Binary impl_info_str should be same only on x64 CPU");
    engine e {engine_kind, 0};

    std::vector<memory::data_type> test_dts {
            memory::data_type::f32, memory::data_type::s8};

    if (!unsupported_data_type(memory::data_type::bf16))
        test_dts.emplace_back(memory::data_type::bf16);

    for (auto dt : test_dts) {
        const auto &binary_tensor_dims = std::get<0>(GetParam());
        const auto format_tag = std::get<2>(GetParam());

        const memory::desc src_0_md {binary_tensor_dims, dt, format_tag};
        const memory::desc src_1_md {binary_tensor_dims, dt, format_tag};
        const memory::desc dst_md {binary_tensor_dims, dt, format_tag};

        const auto binary_desc = binary::desc(
                algorithm::binary_mul, src_0_md, src_1_md, dst_md);
        std::string impl_info_no_postops;

        auto pd = binary::primitive_desc(binary_desc, e);
        ASSERT_NO_THROW(impl_info_no_postops = pd.impl_info_str(););

        dnnl::primitive_attr attr;
        const float scale = 1.f;
        const float alpha = 1.f;
        const float beta = 1.f;
        dnnl::post_ops ops;

        ops.append_sum(1.0);

        ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);

        const auto &binary_po_tensor_dims = std::get<1>(GetParam());
        memory::desc src1_po_md(
                binary_po_tensor_dims, data_type::f32, format_tag);
        ops.append_binary(algorithm::binary_add, src1_po_md);

        attr.set_post_ops(ops);

        std::string impl_info_with_postops;

        pd = binary::primitive_desc(binary_desc, attr, e);
        ASSERT_NO_THROW(impl_info_with_postops = pd.impl_info_str(););
        ASSERT_EQ(impl_info_no_postops, impl_info_with_postops);
    }
}

INSTANTIATE_TEST_SUITE_P(BinaryTensorDims, binary_attr_test_t,
        ::testing::Values(
                // {{src0, src1, dst same_dim}, { binary post-op dim }}
                std::make_tuple(memory::dims {1, 1024}, memory::dims {1, 1024},
                        memory::format_tag::ab),
                std::make_tuple(memory::dims {1, 1024, 1},
                        memory::dims {1, 1024, 1}, memory::format_tag::abc),
                std::make_tuple(memory::dims {1, 1024, 17},
                        memory::dims {1, 1024, 1}, memory::format_tag::abc),
                std::make_tuple(memory::dims {10, 1024, 17, 17},
                        memory::dims {1, 1024, 1, 1},
                        memory::format_tag::abcd)));

static auto expected_failures = []() {
    return ::testing::Values(
            // not supported alg_kind
            binary_test_params_t {{tag::nchw, tag::nchw}, tag::nchw,
                    algorithm::eltwise_relu, {1, 8, 4, 4}, true,
                    dnnl_invalid_arguments},
            // negative dim
            binary_test_params_t {{tag::nchw, tag::nchw}, tag::nchw,
                    algorithm::binary_div, {-1, 8, 4, 4}, true,
                    dnnl_invalid_arguments});
};

static auto zero_dim = []() {
    return ::testing::Values(
            binary_test_params_t {{tag::nchw, tag::nchw}, tag::nchw,
                    algorithm::binary_add, {0, 7, 6, 5}},
            binary_test_params_t {{tag::nChw8c, tag::nhwc}, tag::nChw8c,
                    algorithm::binary_mul, {5, 0, 7, 6}},
            binary_test_params_t {{tag::nChw16c, tag::nchw}, tag::nChw16c,
                    algorithm::binary_div, {8, 15, 0, 5}},
            binary_test_params_t {{tag::nhwc, tag::nChw16c}, tag::nhwc,
                    algorithm::binary_mul, {5, 16, 7, 0}},
            binary_test_params_t {{tag::nhwc, tag::nChw16c}, tag::nhwc,
                    algorithm::binary_sub, {4, 0, 7, 5}},
            binary_test_params_t {{tag::nhwc, tag::nChw16c}, tag::nhwc,
                    algorithm::binary_ge, {4, 16, 7, 0}},
            binary_test_params_t {{tag::nhwc, tag::nChw16c}, tag::nhwc,
                    algorithm::binary_gt, {4, 16, 7, 0}},
            binary_test_params_t {{tag::nhwc, tag::nChw16c}, tag::nhwc,
                    algorithm::binary_le, {4, 16, 7, 0}},
            binary_test_params_t {{tag::nhwc, tag::nChw16c}, tag::nhwc,
                    algorithm::binary_lt, {4, 16, 7, 0}},
            binary_test_params_t {{tag::nhwc, tag::nChw16c}, tag::nhwc,
                    algorithm::binary_eq, {4, 16, 7, 0}},
            binary_test_params_t {{tag::nhwc, tag::nChw16c}, tag::nhwc,
                    algorithm::binary_ne, {4, 16, 7, 0}});
};

static auto simple_cases = []() {
    return ::testing::Values(
            binary_test_params_t {{tag::nchw, tag::nchw}, tag::nchw,
                    algorithm::binary_add, {8, 7, 6, 5}},
            binary_test_params_t {{tag::nhwc, tag::nhwc}, tag::nhwc,
                    algorithm::binary_mul, {5, 8, 7, 6}},
            binary_test_params_t {{tag::nChw8c, tag::nchw}, tag::nChw8c,
                    algorithm::binary_max, {8, 15, 6, 5}},
            binary_test_params_t {{tag::nhwc, tag::nChw16c}, tag::any,
                    algorithm::binary_min, {5, 16, 7, 6}},
            binary_test_params_t {{tag::nchw, tag::nChw16c}, tag::any,
                    algorithm::binary_div, {5, 16, 8, 7}},
            binary_test_params_t {{tag::nchw, tag::nChw16c}, tag::any,
                    algorithm::binary_sub, {5, 16, 8, 7}},
            binary_test_params_t {{tag::nchw, tag::nChw16c}, tag::any,
                    algorithm::binary_ge, {5, 16, 8, 7}},
            binary_test_params_t {{tag::nchw, tag::nChw16c}, tag::any,
                    algorithm::binary_gt, {5, 16, 8, 7}},
            binary_test_params_t {{tag::nchw, tag::nChw16c}, tag::any,
                    algorithm::binary_le, {5, 16, 8, 7}},
            binary_test_params_t {{tag::nchw, tag::nChw16c}, tag::any,
                    algorithm::binary_lt, {5, 16, 8, 7}},
            binary_test_params_t {{tag::nchw, tag::nChw16c}, tag::any,
                    algorithm::binary_eq, {5, 16, 8, 7}},
            binary_test_params_t {{tag::nchw, tag::nChw16c}, tag::any,
                    algorithm::binary_ne, {5, 16, 8, 7}});
};

#define INST_TEST_CASE(test) \
    TEST_P(test, Testsbinary) {} \
    INSTANTIATE_TEST_SUITE_P(TestbinaryEF, test, expected_failures()); \
    INSTANTIATE_TEST_SUITE_P(TestbinaryZero, test, zero_dim()); \
    INSTANTIATE_TEST_SUITE_P(TestbinarySimple, test, simple_cases());

using binary_test_f32 = binary_test_t<float>;
using binary_test_bf16 = binary_test_t<bfloat16_t>;
using binary_test_f16 = binary_test_t<float16_t>;
using binary_test_s8 = binary_test_t<int8_t>;
using binary_test_u8 = binary_test_t<uint8_t>;
using binary_test_s8u8s8 = binary_test_t<int8_t, uint8_t, int8_t>;
using binary_test_u8s8u8 = binary_test_t<uint8_t, int8_t, uint8_t>;
using binary_test_u8s8s8 = binary_test_t<uint8_t, int8_t, int8_t>;
using binary_test_s8u8u8 = binary_test_t<int8_t, uint8_t, uint8_t>;
using binary_test_s8f32u8 = binary_test_t<int8_t, float, uint8_t>;
using binary_test_s8f32s8 = binary_test_t<int8_t, float, int8_t>;
using binary_test_f32u8s8 = binary_test_t<float, uint8_t, int8_t>;
using binary_test_f32f32u8 = binary_test_t<float, float, uint8_t>;

INST_TEST_CASE(binary_test_f32)
INST_TEST_CASE(binary_test_bf16)
INST_TEST_CASE(binary_test_f16)
INST_TEST_CASE(binary_test_s8)
INST_TEST_CASE(binary_test_u8)
INST_TEST_CASE(binary_test_s8u8s8)
INST_TEST_CASE(binary_test_u8s8u8)
INST_TEST_CASE(binary_test_u8s8s8)
INST_TEST_CASE(binary_test_s8u8u8)
INST_TEST_CASE(binary_test_s8f32u8)
INST_TEST_CASE(binary_test_s8f32s8)
INST_TEST_CASE(binary_test_f32u8s8)
INST_TEST_CASE(binary_test_f32f32u8)

} // namespace dnnl
