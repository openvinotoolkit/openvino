// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/range.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/permute.hpp>

namespace cldnn {
// For gtest NE compare, class defines only `==` operator. Required when building using C++20
inline bool operator!=(const range& lhs, const fully_connected& rhs) {
    return !(lhs.operator==(rhs));
}
}  // namespace cldnn

using namespace cldnn;
using namespace ::tests;

TEST(primitive_comparison, common_params) {
    auto def_inputs = {input_info("input0"), input_info("input1"), input_info("input2")};
    auto def_shape = ov::PartialShape{1, 2, 3, 4};
    auto def_data_type = data_types::f32;
    auto def_format = format::bfyx;
    auto def_padding = padding({1, 1, 1, 1});

    auto fc_prim = fully_connected("fc", input_info("input"), "weights");

    auto range_prim = range("range", def_inputs, layout{def_shape, def_data_type, def_format, def_padding});

    auto range_prim_inputs = range("range", {input_info("input0"), input_info("input1")}, layout{def_shape, def_data_type, def_format, def_padding});

    auto range_prim_data_type = range("range", def_inputs, layout{def_shape, data_types::f16, def_format, def_padding});

    ASSERT_NE(range_prim, fc_prim);
    ASSERT_NE(range_prim, range_prim_inputs);
    ASSERT_NE(range_prim, range_prim_data_type);
}

TEST(primitive_comparison, convolution) {
    auto conv_prim = convolution("conv", input_info("input"), "weights", "bias", 1,
                                 {2, 2}, {1, 1}, {0, 0}, {0, 0}, false);

    auto conv_prim_eq = convolution("conv_eq", input_info("input_eq"), "weights_eq", "bias_eq", 1,
                                    {2, 2}, {1, 1}, {0, 0}, {0, 0}, false);

    auto conv_prim_stride = convolution("conv", input_info("input"), "weights", "bias", 1,
                                        {1, 1}, {1, 1}, {0, 0}, {0, 0}, false);

    auto conv_prim_no_bias = convolution("conv", input_info("input"), {"weights"}, {}, 1,
                                         {2, 2}, {1, 1}, {0, 0}, {0, 0}, false);

    auto conv_prim_grouped = convolution("conv", input_info("input"), "weights", "bias", 2,
                                         {2, 2}, {1, 1}, {0, 0}, {0, 0}, true);

    ASSERT_EQ(conv_prim, conv_prim_eq);
    ASSERT_NE(conv_prim, conv_prim_stride);
    ASSERT_NE(conv_prim, conv_prim_no_bias);
    ASSERT_NE(conv_prim, conv_prim_grouped);
}

TEST(primitive_comparison, gemm) {
    auto def_inputs = {input_info("input0"), input_info("input1")};

    auto gemm_prim = gemm("gemm", def_inputs, data_types::f32);
    auto gemm_prim_eq = gemm("gemm_eq", {input_info("input0_eq"), input_info("input1_eq")}, data_types::f32);
    auto gemm_prim_rank = gemm("gemm", def_inputs, data_types::f32, false, false, 1.0f, 0.0f, 2, 2);
    auto gemm_prim_alpha = gemm("gemm", def_inputs, data_types::f32, false, false, 1.5f);
    auto gemm_prim_transpose = gemm("gemm", def_inputs, data_types::f32, true, false);

    ASSERT_EQ(gemm_prim, gemm_prim_eq);
    ASSERT_NE(gemm_prim, gemm_prim_rank);
    ASSERT_NE(gemm_prim, gemm_prim_alpha);
    ASSERT_NE(gemm_prim, gemm_prim_transpose);
}

TEST(primitive_comparison, fully_connected) {
    auto fc_prim = fully_connected("fc", input_info("input"), "weights", "bias", 2);
    auto fc_prim_eq = fully_connected("fc_eq", input_info("input_eq"), "weights_eq", "bias_eq", 2);
    auto fc_prim_bias = fully_connected("fc", input_info("input"), "weights", "", 2);
    auto fc_prim_input_size = fully_connected("fc", input_info("input"), "weights", "bias", 4);

    ASSERT_EQ(fc_prim, fc_prim_eq);
    ASSERT_NE(fc_prim, fc_prim_bias);
    ASSERT_NE(fc_prim, fc_prim_input_size);
}

TEST(primitive_comparison, gather) {
    auto gather_prim = gather("gather", input_info("input0"), input_info("input1"), 2, {}, {1, 3, 224, 224}, 1, true);
    auto gather_prim_eq = gather("gather_eq", input_info("input0_eq"), input_info("input1_eq"), 2, {}, {1, 3, 224, 224}, 1, true);
    auto gather_prim_axis = gather("gather", input_info("input0"), input_info("input1"), 3, {}, {1, 3, 224, 224}, 1, true);
    auto gather_prim_batch_dim = gather("gather", input_info("input0"), input_info("input1"), 2, {}, {1, 3, 224, 224}, 2, true);
    auto gather_prim_support_neg_ind = gather("gather", input_info("input0"), input_info("input1"), 2, {}, {1, 3, 224, 224}, 1, false);

    ASSERT_EQ(gather_prim, gather_prim_eq);
    ASSERT_NE(gather_prim, gather_prim_axis);
    ASSERT_NE(gather_prim, gather_prim_batch_dim);
    ASSERT_NE(gather_prim, gather_prim_support_neg_ind);
}

TEST(primitive_comparison, permute) {
    auto permute_prim = permute("permute", input_info("input"), {0, 1, 2, 3});
    auto permute_prim_eq = permute("permute_eq", input_info("input_eq"), {0, 1, 2, 3});
    auto permute_prim_order = permute("permute", input_info("input"), {3, 2, 1, 0});

    ASSERT_EQ(permute_prim, permute_prim_eq);
    ASSERT_NE(permute_prim, permute_prim_order);
}

TEST(primitive_comparison, reorder_weights) {
    auto shape = ov::PartialShape{1, 2, 3, 4};
    auto data_type = data_types::f32;

    auto format_osv16 = format::os_iyx_osv16;
    auto format_osv32 = format::os_iyx_osv32;

    auto layout_osv16 = layout{shape, data_type, format_osv16};
    auto layout_osv32 = layout{shape, data_type, format_osv32};

    auto reorder_weights_prim = reorder("reorder_weights", input_info("input"), layout_osv16);
    auto reorder_weights_eq_prim = reorder("reorder_weights_eq", input_info("input"), layout_osv16);
    auto reorder_weights_diff_prim = reorder("reorder_weights_neq", input_info("input"), layout_osv32);

    ASSERT_EQ(reorder_weights_prim, reorder_weights_eq_prim);
    ASSERT_NE(reorder_weights_prim, reorder_weights_diff_prim);
}
