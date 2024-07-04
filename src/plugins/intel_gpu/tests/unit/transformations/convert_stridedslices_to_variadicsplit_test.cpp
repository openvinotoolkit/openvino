// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/variadic_split.hpp"

#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

#include "plugin/transformations/convert_stridedslices_to_variadicsplit.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, ConvertStridedSlicesToVariadicSplit1) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 3072});
        auto weight_const = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{9216, 3072});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{9216, 24});
        auto zp_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1});
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_const, no_bias, scale_const, zp_const);
        auto begin_const1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, 0});
        auto end_const1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, 3072});
        auto strides_const1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 1, 1});
        auto strided_slice1 = std::make_shared<ov::op::v1::StridedSlice>(fc_compressed,
                                                                         begin_const1,
                                                                         end_const1,
                                                                         strides_const1,
                                                                         std::vector<std::int64_t>{1, 1, 0},
                                                                         std::vector<std::int64_t>{1, 1, 0});
        auto begin_const2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, 3072});
        auto end_const2= ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, 6144});
        auto strides_const2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 1, 1});
        auto strided_slice2 = std::make_shared<ov::op::v1::StridedSlice>(fc_compressed,
                                                                         begin_const2,
                                                                         end_const2,
                                                                         strides_const2,
                                                                         std::vector<std::int64_t>{1, 1, 0},
                                                                         std::vector<std::int64_t>{1, 1, 0});
        auto begin_const3 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, 6144});
        auto end_const3 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, 9216});
        auto strides_const3 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 1, 1});
        auto strided_slice3 = std::make_shared<ov::op::v1::StridedSlice>(fc_compressed,
                                                                         begin_const3,
                                                                         end_const3,
                                                                         strides_const3,
                                                                         std::vector<std::int64_t>{1, 1, 0},
                                                                         std::vector<std::int64_t>{1, 1, 0});
        auto shape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 0, 32, 96});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(strided_slice1, shape_const, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(strided_slice2, shape_const, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(strided_slice3, shape_const, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);

        model = std::make_shared<ov::Model>(ov::ResultVector{ result1, result2, result3 }, ov::ParameterVector{ input });
        manager.register_pass<ConvertStridedSlicesToVariadicSplit>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 3072});
        auto weight_const = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{9216, 3072});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{9216, 24});
        auto zp_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1});
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_const, no_bias, scale_const, zp_const);
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{3072, 3072, 3072});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(fc_compressed, axis_const, split_lengths_const);
        auto shape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 0, 32, 96});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(variadic_split->output(0), shape_const, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(variadic_split->output(1), shape_const, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(variadic_split->output(2), shape_const, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{ result1, result2, result3 }, ov::ParameterVector{ input });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, ConvertStridedSlicesToVariadicSplit2) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 3072});
        auto weight_const = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{9216, 3072});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{9216, 24});
        auto zp_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1});
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_const, no_bias, scale_const, zp_const);
        auto begin_const1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, 0});
        auto end_const1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 3072});
        auto strides_const1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 1, 1});
        auto strided_slice1 = std::make_shared<ov::op::v1::StridedSlice>(fc_compressed,
                                                                         begin_const1,
                                                                         end_const1,
                                                                         strides_const1,
                                                                         std::vector<std::int64_t>{1, 0, 0},
                                                                         std::vector<std::int64_t>{1, 0, 0});
        auto begin_const2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 3072});
        auto end_const2= ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 4, 6144});
        auto strides_const2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 1, 1});
        auto strided_slice2 = std::make_shared<ov::op::v1::StridedSlice>(fc_compressed,
                                                                         begin_const2,
                                                                         end_const2,
                                                                         strides_const2,
                                                                         std::vector<std::int64_t>{1, 0, 0},
                                                                         std::vector<std::int64_t>{1, 0, 0});
        auto begin_const3 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 4, 6144});
        auto end_const3 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 7, 9216});
        auto strides_const3 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 1, 1});
        auto strided_slice3 = std::make_shared<ov::op::v1::StridedSlice>(fc_compressed,
                                                                         begin_const3,
                                                                         end_const3,
                                                                         strides_const3,
                                                                         std::vector<std::int64_t>{1, 0, 0},
                                                                         std::vector<std::int64_t>{1, 0, 0});
        auto shape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 0, 32, 96});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(strided_slice1, shape_const, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(strided_slice2, shape_const, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(strided_slice3, shape_const, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);

        model = std::make_shared<ov::Model>(ov::ResultVector{ result1, result2, result3 }, ov::ParameterVector{ input });
        manager.register_pass<ConvertStridedSlicesToVariadicSplit>();
    }
    {
        model_ref = model->clone();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, ConvertStridedSlicesToVariadicSplit3) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 3072});
        auto weight_const = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{9216, 3072});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{9216, 24});
        auto zp_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1});
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_const, no_bias, scale_const, zp_const);
        auto begin_const1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, 0});
        auto end_const1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, 3072});
        auto strides_const1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 1, 1});
        auto strided_slice1 = std::make_shared<ov::op::v1::StridedSlice>(fc_compressed,
                                                                         begin_const1,
                                                                         end_const1,
                                                                         strides_const1,
                                                                         std::vector<std::int64_t>{1, 1, 0},
                                                                         std::vector<std::int64_t>{1, 1, 0});
        auto begin_const2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, 3072});
        auto end_const2= ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, 6144});
        auto strides_const2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 1, 1});
        auto strided_slice2 = std::make_shared<ov::op::v1::StridedSlice>(fc_compressed,
                                                                         begin_const2,
                                                                         end_const2,
                                                                         strides_const2,
                                                                         std::vector<std::int64_t>{1, 1, 0},
                                                                         std::vector<std::int64_t>{1, 1, 0});
        auto shape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 0, 32, 96});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(strided_slice1, shape_const, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(strided_slice2, shape_const, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(fc_compressed);

        model = std::make_shared<ov::Model>(ov::ResultVector{ result1, result2, result3 }, ov::ParameterVector{ input });
        manager.register_pass<ConvertStridedSlicesToVariadicSplit>();
    }
    {
        model_ref = model->clone();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}


}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
