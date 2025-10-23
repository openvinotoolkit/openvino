// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/matrix_nms.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
std::string MatrixNmsLayerTest::getTestCaseName(const testing::TestParamInfo<NmsParams>& obj) {
    const auto& [shapes,
                 model_type,
                 sort_result_type,
                 out_type,
                 top_k_params,
                 threshold_params,
                 backgroudClass,
                 normalized,
                 decayFunction,
                 target_device] = obj.param;

    const auto& [nms_top_k, keep_top_k] = top_k_params;

    const auto& [score_threshold, gaussian_sigma, post_threshold] = threshold_params;

    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }

    using ov::test::utils::operator<<;
    result << ")_model_type=" << model_type << "_";
    result << "sortResultType=" << sort_result_type << "_normalized=" << normalized << "_";
    result << "out_type=" << out_type << "_nms_top_k=" << nms_top_k << "_keep_top_k=" << keep_top_k << "_";
    result << "backgroudClass=" << backgroudClass << "_decayFunction=" << decayFunction << "_";
    result << "score_threshold=" << score_threshold << "_gaussian_sigma=" << gaussian_sigma << "_";
    result << "post_threshold=" << post_threshold <<"_TargetDevice=" << target_device;
    return result.str();
}

void MatrixNmsLayerTest::SetUp() {
    ov::op::v8::MatrixNms::Attributes attrs;

    const auto& [shapes,
                 model_type,
                 _sort_result_type,
                 _output_type,
                 top_k_params,
                 threshold_params,
                 _background_class,
                 _normalized,
                 _decay_function,
                 _targetDevice] = this->GetParam();
    attrs.sort_result_type = _sort_result_type;
    attrs.output_type = _output_type;
    attrs.background_class = _background_class;
    attrs.normalized = _normalized;
    attrs.decay_function = _decay_function;
    targetDevice = _targetDevice;

    std::tie(attrs.nms_top_k, attrs.keep_top_k) = top_k_params;
    std::tie(attrs.score_threshold, attrs.gaussian_sigma, attrs.post_threshold) = threshold_params;

    init_input_shapes(shapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }
    auto nms = std::make_shared<ov::op::v8::MatrixNms>(params[0], params[1], attrs);

    function = std::make_shared<ov::Model>(nms, params, "MatrixNMS");
}
} // namespace test
} // namespace ov
