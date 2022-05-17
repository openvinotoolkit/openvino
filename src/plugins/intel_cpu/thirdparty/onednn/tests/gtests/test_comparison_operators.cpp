/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "dnnl.hpp"

#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {

namespace {
bool self_compare(const dnnl::primitive_attr &attr) {
    return *attr.get() == *attr.get();
}

template <typename T>
bool self_compare(const T &desc) {
    return dnnl::impl::operator==(desc, desc);
}

} // namespace

#define TEST_SELF_COMPARISON(v) ASSERT_EQ(true, self_compare(v))

class comparison_operators_t : public ::testing::Test {};

HANDLE_EXCEPTIONS_FOR_TEST(comparison_operators_t, TestAttrOutputScales) {
    dnnl::primitive_attr attr;

    attr.set_output_scales(0, {NAN});
    TEST_SELF_COMPARISON(attr);

    attr.set_output_scales(1 << 1, {1.5, NAN, 3.5});
    TEST_SELF_COMPARISON(attr);
}

HANDLE_EXCEPTIONS_FOR_TEST(comparison_operators_t, TestAttrArgScales) {
    dnnl::primitive_attr attr;

    attr.set_scales(DNNL_ARG_SRC_0, 0, {NAN});
    TEST_SELF_COMPARISON(attr);

    attr.set_scales(DNNL_ARG_SRC_0, 1 << 1, {1.5f, NAN, 3.5f});
    TEST_SELF_COMPARISON(attr);
}

TEST(comparison_operators_t, TestAttrDataQparams) {
    dnnl::primitive_attr attr;

    attr.set_rnn_data_qparams(1.5f, NAN);
    TEST_SELF_COMPARISON(attr);
}

HANDLE_EXCEPTIONS_FOR_TEST(comparison_operators_t, TestAttrWeightsQparams) {
    dnnl::primitive_attr attr;

    attr.set_rnn_weights_qparams(0, {NAN});
    TEST_SELF_COMPARISON(attr);

    attr.set_rnn_weights_qparams(1 << 1, {1.5f, NAN, 3.5f});
    TEST_SELF_COMPARISON(attr);
}

HANDLE_EXCEPTIONS_FOR_TEST(
        comparison_operators_t, TestAttrWeightsProjectionQparams) {
    dnnl::primitive_attr attr;

    attr.set_rnn_weights_projection_qparams(0, {NAN});
    TEST_SELF_COMPARISON(attr);

    attr.set_rnn_weights_projection_qparams(1 << 1, {1.5f, NAN, 3.5f});
    TEST_SELF_COMPARISON(attr);
}

TEST(comparison_operators_t, TestSumPostOp) {
    dnnl::primitive_attr attr;
    dnnl::post_ops ops;

    ops.append_sum(NAN);
    attr.set_post_ops(ops);
    TEST_SELF_COMPARISON(attr);
}

TEST(comparison_operators_t, TestEltwisePostOp) {
    dnnl::primitive_attr attr;
    dnnl::post_ops ops;

    ops.append_eltwise(NAN, algorithm::eltwise_bounded_relu, 2.5f, 3.5f);
    attr.set_post_ops(ops);
    TEST_SELF_COMPARISON(attr);
}

HANDLE_EXCEPTIONS_FOR_TEST(comparison_operators_t, TestDepthwisePostOp) {
    dnnl::primitive_attr attr;
    dnnl::post_ops ops;

    ops.append_dw_k3s1p1(memory::data_type::s8, memory::data_type::f32,
            memory::data_type::u8, 0, {NAN});
    attr.set_post_ops(ops);
    TEST_SELF_COMPARISON(attr);

    ops.append_dw_k3s2p1(memory::data_type::u8, memory::data_type::s32,
            memory::data_type::f32, 1 << 1, {1.5f, NAN, 3.5f});
    attr.set_post_ops(ops);
    TEST_SELF_COMPARISON(attr);
}

TEST(comparison_operators_t, TestBatchNormDesc) {
    auto bnorm_desc = dnnl_batch_normalization_desc_t();
    bnorm_desc.batch_norm_epsilon = NAN;
    TEST_SELF_COMPARISON(bnorm_desc);
}

TEST(comparison_operators_t, TestEltwiseDesc) {
    auto eltwise_desc = dnnl_eltwise_desc_t();
    eltwise_desc.alpha = NAN;
    TEST_SELF_COMPARISON(eltwise_desc);
}

TEST(comparison_operators_t, TestLayerNormDesc) {
    auto lnorm_desc = dnnl_layer_normalization_desc_t();
    lnorm_desc.layer_norm_epsilon = NAN;
    TEST_SELF_COMPARISON(lnorm_desc);
}

TEST(comparison_operators_t, TestLRNDesc) {
    auto lrn_desc = dnnl_lrn_desc_t();
    lrn_desc.lrn_alpha = NAN;
    TEST_SELF_COMPARISON(lrn_desc);
}

TEST(comparison_operators_t, TestReductionDesc) {
    auto reduction_desc = dnnl_reduction_desc_t();
    reduction_desc.p = NAN;
    TEST_SELF_COMPARISON(reduction_desc);
}

TEST(comparison_operators_t, TestResamplingDesc) {
    auto resampling_desc = dnnl_resampling_desc_t();
    resampling_desc.factors[0] = NAN;
    TEST_SELF_COMPARISON(resampling_desc);
}

TEST(comparison_operators_t, TestRNNDesc) {
    auto rnn_desc = dnnl_rnn_desc_t();
    rnn_desc.alpha = NAN;
    TEST_SELF_COMPARISON(rnn_desc);
}

TEST(comparison_operators_t, TestSumDesc) {
    float scales[2] = {NAN, 2.5f};
    dnnl_memory_desc_t src_mds[2] = {};
    dnnl_memory_desc_t dst_md {};

    dnnl::impl::dnnl_sum_desc_t sum_desc
            = {dnnl::impl::primitive_kind::sum, &dst_md, 2, scales, src_mds};
    TEST_SELF_COMPARISON(sum_desc);
}

} // namespace dnnl
