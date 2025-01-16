// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <transformations/cpu_opset/common/op/fully_connected.hpp>
#include <transformations/cpu_opset/common/pass/split_fc.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include "openvino/core/visibility.hpp"
#include <openvino/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace testing;
using namespace ov::intel_cpu;

#if defined (OPENVINO_ARCH_ARM) && defined(__linux__)
// Ticket: 153166
TEST_F(TransformationTestsF, DISABLED_SplitFCTest) {
#else
TEST_F(TransformationTestsF, SplitFCTest) {
#endif
    disable_rt_info_check();
    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 3, 4096, 1 });
        auto transpose_constant_src = ov::opset1::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose_src = std::make_shared<ov::opset1::Transpose>(src, transpose_constant_src);

        auto wgt = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 2048, 4096 }, { 12.34 });

        auto fc = std::make_shared<FullyConnectedNode>(transpose_src, wgt, ov::Rank(3));
        model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{src});
        manager.register_pass<SplitFC>(1);
    }
    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 3, 4096, 1 });
        auto transpose_constant_src = ov::opset1::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose_src = std::make_shared<ov::opset1::Transpose>(src, transpose_constant_src);

        auto wgt = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 2048, 4096 }, { 12.34 });

        auto split_dim_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
        auto split_length = ov::opset1::Constant::create<int32_t>(ov::element::i32, ov::Shape{2}, {1024, 1024});
        auto split_wgts = std::make_shared<ov::opset1::VariadicSplit>(wgt, split_dim_node, split_length);

        auto fc0 = std::make_shared<FullyConnectedNode>(transpose_src, split_wgts->output(0), ov::Rank(3));
        auto fc1 = std::make_shared<FullyConnectedNode>(transpose_src, split_wgts->output(1), ov::Rank(3));

        ov::NodeVector concat_args({fc0, fc1});
        constexpr size_t concat_dim = -1;
        auto concat = std::make_shared<ov::opset1::Concat>(concat_args, concat_dim);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{src});
    }
}

#if defined (OPENVINO_ARCH_ARM) && defined(__linux__)
// Ticket: 153166
TEST_F(TransformationTestsF, DISABLED_SplitFCTest_int8_weight) {
#else
TEST_F(TransformationTestsF, SplitFCTest_int8_weight) {
#endif
    disable_rt_info_check();
    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 4096, 1});
        auto transpose_constant_src = ov::opset1::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose_src = std::make_shared<ov::opset1::Transpose>(src, transpose_constant_src);

        auto wgt = ov::opset1::Constant::create(ov::element::u8, ov::Shape{2048, 4096}, {123});
        auto cvt_wgt = std::make_shared<ov::opset1::Convert>(wgt, ov::element::f32);

        auto zp = ov::opset1::Constant::create(ov::element::u8, ov::Shape{2048, 1}, {1});
        auto cvt_zp = std::make_shared<ov::opset1::Convert>(zp, ov::element::f32);

        auto sub = std::make_shared<ov::opset1::Subtract>(cvt_wgt, cvt_zp);

        auto mul_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2048, 1}, {0.2});
        auto mul = std::make_shared<ov::opset1::Multiply>(sub, mul_const);

        auto fc = std::make_shared<FullyConnectedNode>(transpose_src, mul, ov::Rank(3));
        model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{src});
        manager.register_pass<SplitFC>(1);
    }
    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 3, 4096, 1 });
        auto transpose_constant_src = ov::opset1::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose_src = std::make_shared<ov::opset1::Transpose>(src, transpose_constant_src);

        auto wgt = ov::opset1::Constant::create(ov::element::u8, ov::Shape{ 2048, 4096 }, { 123 });
        auto cvt_wgt = std::make_shared<ov::opset1::Convert>(wgt, ov::element::f32);

        auto split_dim_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
        auto split_length = ov::opset1::Constant::create<int32_t>(ov::element::i32, ov::Shape{2}, {1024, 1024});

        auto split_wgts = std::make_shared<ov::opset1::VariadicSplit>(wgt, split_dim_node, split_length);
        auto cvt_wgt0 = std::make_shared<ov::opset1::Convert>(split_wgts->output(0), ov::element::f32);
        auto cvt_wgt1 = std::make_shared<ov::opset1::Convert>(split_wgts->output(1), ov::element::f32);

        auto zp = ov::opset1::Constant::create(ov::element::u8, ov::Shape{2048, 1}, {1});
        auto split_zp = std::make_shared<ov::opset1::VariadicSplit>(zp, split_dim_node, split_length);

        auto cvt_zp0 = std::make_shared<ov::opset1::Convert>(split_zp->output(0), ov::element::f32);
        auto cvt_zp1 = std::make_shared<ov::opset1::Convert>(split_zp->output(1), ov::element::f32);

        auto sub0 = std::make_shared<ov::opset1::Subtract>(cvt_wgt0, cvt_zp0);
        auto sub1 = std::make_shared<ov::opset1::Subtract>(cvt_wgt1, cvt_zp1);

        auto mul_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2048, 1}, {0.2});
        auto split_mul_const = std::make_shared<ov::opset1::VariadicSplit>(mul_const, split_dim_node, split_length);

        auto mul0 = std::make_shared<ov::opset1::Multiply>(sub0, split_mul_const->output(0));
        auto mul1 = std::make_shared<ov::opset1::Multiply>(sub1, split_mul_const->output(1));

        auto fc0 = std::make_shared<FullyConnectedNode>(transpose_src, mul0, ov::Rank(3));
        auto fc1 = std::make_shared<FullyConnectedNode>(transpose_src, mul1, ov::Rank(3));

        ov::NodeVector concat_args({fc0, fc1});
        constexpr size_t concat_dim = -1;
        auto concat = std::make_shared<ov::opset1::Concat>(concat_args, concat_dim);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{src});
    }
}

#if defined (OPENVINO_ARCH_ARM) && defined(__linux__)
// Ticket: 153166
TEST_F(TransformationTestsF, DISABLED_SplitFCTest_int4_weight) {
#else
TEST_F(TransformationTestsF, SplitFCTest_int4_weight) {
#endif
    disable_rt_info_check();
    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 4096, 1});
        auto transpose_constant_src = ov::opset1::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose_src = std::make_shared<ov::opset1::Transpose>(src, transpose_constant_src);

        auto wgt = ov::opset1::Constant::create(ov::element::u4, ov::Shape{2048, 4096}, {12});
        auto cvt_wgt = std::make_shared<ov::opset1::Convert>(wgt, ov::element::f32);

        auto zp = ov::opset1::Constant::create(ov::element::u4, ov::Shape{2048, 1}, {1});
        auto cvt_zp = std::make_shared<ov::opset1::Convert>(zp, ov::element::f32);

        auto sub = std::make_shared<ov::opset1::Subtract>(cvt_wgt, cvt_zp);

        auto mul_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2048, 1}, {0.2});
        auto mul = std::make_shared<ov::opset1::Multiply>(sub, mul_const);

        auto fc = std::make_shared<FullyConnectedNode>(transpose_src, mul, ov::Rank(3));
        model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{src});
        manager.register_pass<SplitFC>(1);
    }
    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 4096, 1});
        auto transpose_constant_src = ov::opset1::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose_src = std::make_shared<ov::opset1::Transpose>(src, transpose_constant_src);

        auto wgt = ov::opset1::Constant::create(ov::element::u4, ov::Shape{2048, 4096}, {12});
        auto cvt_wgt_i8 = std::make_shared<ov::opset1::Convert>(wgt, ov::element::i8);

        auto split_dim_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
        auto split_length = ov::opset1::Constant::create<int32_t>(ov::element::i32, ov::Shape{2}, {1024, 1024});

        auto split_wgts = std::make_shared<ov::opset1::VariadicSplit>(cvt_wgt_i8, split_dim_node, split_length);
        auto cvt_wgt0_u4 = std::make_shared<ov::opset1::Convert>(split_wgts->output(0), ov::element::u4);
        auto cvt_wgt1_u4 = std::make_shared<ov::opset1::Convert>(split_wgts->output(1), ov::element::u4);
        auto cvt_wgt0_f32 = std::make_shared<ov::opset1::Convert>(cvt_wgt0_u4, ov::element::f32);
        auto cvt_wgt1_f32 = std::make_shared<ov::opset1::Convert>(cvt_wgt1_u4, ov::element::f32);

        auto zp = ov::opset1::Constant::create(ov::element::u4, ov::Shape{2048, 1}, {1});
        auto cvt_zp_i8 = std::make_shared<ov::opset1::Convert>(zp, ov::element::i8);
        auto split_zp = std::make_shared<ov::opset1::VariadicSplit>(cvt_zp_i8, split_dim_node, split_length);

        auto cvt_zp0_u4 = std::make_shared<ov::opset1::Convert>(split_zp->output(0), ov::element::u4);
        auto cvt_zp1_u4 = std::make_shared<ov::opset1::Convert>(split_zp->output(1), ov::element::u4);
        auto cvt_zp0_f32 = std::make_shared<ov::opset1::Convert>(cvt_zp0_u4, ov::element::f32);
        auto cvt_zp1_f32 = std::make_shared<ov::opset1::Convert>(cvt_zp1_u4, ov::element::f32);

        auto sub0 = std::make_shared<ov::opset1::Subtract>(cvt_wgt0_f32, cvt_zp0_f32);
        auto sub1 = std::make_shared<ov::opset1::Subtract>(cvt_wgt1_f32, cvt_zp1_f32);

        auto mul_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2048, 1}, {0.2});
        auto split_mul_const = std::make_shared<ov::opset1::VariadicSplit>(mul_const, split_dim_node, split_length);

        auto mul0 = std::make_shared<ov::opset1::Multiply>(sub0, split_mul_const->output(0));
        auto mul1 = std::make_shared<ov::opset1::Multiply>(sub1, split_mul_const->output(1));

        auto fc0 = std::make_shared<FullyConnectedNode>(transpose_src, mul0, ov::Rank(3));
        auto fc1 = std::make_shared<FullyConnectedNode>(transpose_src, mul1, ov::Rank(3));

        ov::NodeVector concat_args({fc0, fc1});
        constexpr size_t concat_dim = -1;
        auto concat = std::make_shared<ov::opset1::Concat>(concat_args, concat_dim);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{src});
    }
}

#if (defined OPENVINO_ARCH_ARM && defined(__linux__))
// Ticket: 153166
TEST_F(TransformationTestsF, DISABLED_SplitFCTest_int4_weight_reshape) {
#else
TEST_F(TransformationTestsF, SplitFCTest_int4_weight_reshape) {
#endif
    disable_rt_info_check();
    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 3, 2048, 1 });
        auto transpose_constant_src = ov::opset1::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose_src = std::make_shared<ov::opset1::Transpose>(src, transpose_constant_src);

        auto wgt = ov::opset1::Constant::create(ov::element::u4, ov::Shape{ 4096, 2, 1024}, { 12 });
        auto cvt_wgt = std::make_shared<ov::opset1::Convert>(wgt, ov::element::f32);

        auto zp = ov::opset1::Constant::create(ov::element::u4, ov::Shape{1}, { 1 });
        auto cvt_zp = std::make_shared<ov::opset1::Convert>(zp, ov::element::f32);

        auto sub = std::make_shared<ov::opset1::Subtract>(cvt_wgt, cvt_zp);

        auto mul_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{4096, 2, 1}, {0.2});
        auto mul = std::make_shared<ov::opset1::Multiply>(sub, mul_const);

        auto res_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{2}, {4096, 2048});
        auto reshape = std::make_shared<ov::opset1::Reshape>(mul, res_const, false);

        auto fc = std::make_shared<FullyConnectedNode>(transpose_src, reshape, ov::Rank(3));
        model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{src});
        manager.register_pass<SplitFC>(1);
    }
    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 3, 2048, 1 });
        auto transpose_constant_src = ov::opset1::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose_src = std::make_shared<ov::opset1::Transpose>(src, transpose_constant_src);

        auto wgt = ov::opset1::Constant::create(ov::element::u4, ov::Shape{ 4096, 2, 1024 }, { 12 });
        auto cvt_wgt_i8 = std::make_shared<ov::opset1::Convert>(wgt, ov::element::i8);

        auto split_dim_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
        auto split_length = ov::opset1::Constant::create<int32_t>(ov::element::i32, ov::Shape{2}, {2048, 2048});

        auto split_wgts = std::make_shared<ov::opset1::VariadicSplit>(cvt_wgt_i8, split_dim_node, split_length);
        auto cvt_wgt0_u4 = std::make_shared<ov::opset1::Convert>(split_wgts->output(0), ov::element::u4);
        auto cvt_wgt1_u4 = std::make_shared<ov::opset1::Convert>(split_wgts->output(1), ov::element::u4);
        auto cvt_wgt0_f32 = std::make_shared<ov::opset1::Convert>(cvt_wgt0_u4, ov::element::f32);
        auto cvt_wgt1_f32 = std::make_shared<ov::opset1::Convert>(cvt_wgt1_u4, ov::element::f32);

        auto zp = ov::opset1::Constant::create(ov::element::u4, ov::Shape{1}, { 1 });
        auto zp0 = std::make_shared<ov::opset1::Constant>(zp->get_element_type(), zp->get_shape(), zp->get_data_ptr());
        auto zp1 = std::make_shared<ov::opset1::Constant>(zp->get_element_type(), zp->get_shape(), zp->get_data_ptr());

        auto cvt_zp0 = std::make_shared<ov::opset1::Convert>(zp0, ov::element::f32);
        auto cvt_zp1 = std::make_shared<ov::opset1::Convert>(zp1, ov::element::f32);

        auto sub0 = std::make_shared<ov::opset1::Subtract>(cvt_wgt0_f32, cvt_zp0);
        auto sub1 = std::make_shared<ov::opset1::Subtract>(cvt_wgt1_f32, cvt_zp1);

        auto mul_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{4096, 2, 1}, {0.2});
        auto split_mul_const = std::make_shared<ov::opset1::VariadicSplit>(mul_const, split_dim_node, split_length);

        auto mul0 = std::make_shared<ov::opset1::Multiply>(sub0, split_mul_const->output(0));
        auto mul1 = std::make_shared<ov::opset1::Multiply>(sub1, split_mul_const->output(1));

        std::vector<int32_t> reshape_pattern_vec = {2048, 2048};
        auto reshape_pattern = std::make_shared<ov::opset1::Constant>(ov::element::i32, ov::Shape{2}, reshape_pattern_vec);
        auto reshape0 = std::make_shared<ov::opset1::Reshape>(mul0, reshape_pattern, false);
        auto reshape1 = std::make_shared<ov::opset1::Reshape>(mul1, reshape_pattern, false);

        auto fc0 = std::make_shared<FullyConnectedNode>(transpose_src, reshape0, ov::Rank(3));
        auto fc1 = std::make_shared<FullyConnectedNode>(transpose_src, reshape1, ov::Rank(3));

        ov::NodeVector concat_args({fc0, fc1});
        constexpr size_t concat_dim = -1;
        auto concat = std::make_shared<ov::opset1::Concat>(concat_args, concat_dim);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{src});
    }
}
