// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/normalize_l2_decomposition.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/manager.hpp"

namespace ov::test::intel_gpu {

TEST_F(TransformationTestsF, NormalizeL2LastAxisToFeatureRank3) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, 3, 4});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto normalize = std::make_shared<ov::op::v0::NormalizeL2>(input, axes, 1e-6, ov::op::EpsMode::ADD);

    model = std::make_shared<ov::Model>(ov::OutputVector{normalize}, ov::ParameterVector{input});
    manager.register_pass<ov::intel_gpu::NormalizeL2LastAxisToFeature>();

    auto feature_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {6, 4, 1, 1});
    auto to_feature = std::make_shared<ov::op::v1::Reshape>(input, feature_shape, false);
    auto feature_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto feature_normalize = std::make_shared<ov::op::v0::NormalizeL2>(
        to_feature, feature_axis, 1e-6, ov::op::EpsMode::ADD);
    auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {2, 3, 4});
    auto restore_shape = std::make_shared<ov::op::v1::Reshape>(feature_normalize, output_shape, false);

    model_ref = std::make_shared<ov::Model>(ov::OutputVector{restore_shape}, ov::ParameterVector{input});
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

}  // namespace ov::test::intel_gpu