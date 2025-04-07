// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/detection_output.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/tile.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_detection_output(ov::element::Type type) {
    const auto& data = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{1, 4, 10, 10});

    const auto& constant_0 = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{1, 1, 1, 1});
    const auto& mul_0 = std::make_shared<ov::op::v1::Multiply>(data, constant_0);

    const auto& filters = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{1, 4, 1, 1});
    const auto& conv = std::make_shared<ov::op::v1::Convolution>(mul_0,
                                                                 filters,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});

    const auto& box_logits_reshape =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, -1});
    const auto& box_logits = std::make_shared<ov::op::v1::Reshape>(conv, box_logits_reshape, true);

    const auto& four_times = std::make_shared<ov::op::v0::Tile>(
        box_logits,
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 4}));

    const auto& third_input_reshape =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, -1});
    const auto& third_input = std::make_shared<ov::op::v1::Reshape>(four_times, third_input_reshape, true);

    ov::op::v0::DetectionOutput::Attributes attr;
    attr.num_classes = 4;
    attr.background_label_id = 0;
    attr.top_k = 75;
    attr.variance_encoded_in_target = true;
    attr.keep_top_k = {50};
    attr.code_type = std::string{"caffe.PriorBoxParameter.CORNER"};
    attr.share_location = true;
    attr.nms_threshold = 0.5f;
    attr.confidence_threshold = 0.5f;
    attr.clip_after_nms = false;
    attr.clip_before_nms = false;
    attr.decrease_label_id = false;
    attr.normalized = true;
    attr.input_height = 1;
    attr.input_width = 1;
    attr.objectness_score = 0.4f;
    const auto& detection = std::make_shared<ov::op::v0::DetectionOutput>(four_times, four_times, third_input, attr);
    const auto& convert = std::make_shared<ov::op::v0::Convert>(detection, type);

    return std::make_shared<ov::Model>(ov::NodeVector{convert}, ov::ParameterVector{data}, "SplitableDetectionOutput");
}
}  // namespace utils
}  // namespace test
}  // namespace ov