// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/matrix_nms.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
std::string MatrixNmsLayerTest::getTestCaseName(const testing::TestParamInfo<NmsParams>& obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    op::v8::MatrixNms::SortResultType sort_result_type;
    ov::element::Type out_type;
    int backgroudClass;
    op::v8::MatrixNms::DecayFunction decayFunction;
    TopKParams top_k_params;
    ThresholdParams threshold_params;
    bool normalized;
    std::string target_device;
    std::tie(shapes, model_type, sort_result_type, out_type, top_k_params, threshold_params,
        backgroudClass, normalized, decayFunction, target_device) = obj.param;

    int nms_top_k, keep_top_k;
    std::tie(nms_top_k, keep_top_k) = top_k_params;

    float score_threshold, gaussian_sigma, post_threshold;
    std::tie(score_threshold, gaussian_sigma, post_threshold) = threshold_params;

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
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    TopKParams top_k_params;
    ThresholdParams threshold_params;
    ov::op::v8::MatrixNms::Attributes attrs;

    std::tie(shapes, model_type, attrs.sort_result_type, attrs.output_type, top_k_params, threshold_params,
        attrs.background_class, attrs.normalized, attrs.decay_function, targetDevice) = this->GetParam();

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
