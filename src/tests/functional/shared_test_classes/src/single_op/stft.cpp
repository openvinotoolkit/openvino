// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/stft.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"

namespace ov {
namespace test {

std::string STFTLayerTest::getTestCaseName(const testing::TestParamInfo<STFTParams>& obj) {
    std::ostringstream result;
    const std::vector<InputShape>& data_shapes = std::get<0>(obj.param);
    const int64_t frame_size = std::get<1>(obj.param);
    const int64_t frame_step = std::get<2>(obj.param);
    const bool transpose_frames = std::get<3>(obj.param);
    const ElementType& data_type = std::get<4>(obj.param);
    const ElementType& step_size_type = std::get<5>(obj.param);
    const utils::InputLayerType& param_type = std::get<6>(obj.param);
    const ov::test::TargetDevice& dev = std::get<7>(obj.param);

    for (size_t s = 0lu; s < 2; s++) {
        const auto& shape_item = data_shapes[s];
        result << "IS" << s << "=(";
        result << shape_item.first;
        result << ")_TS=";

        for (size_t i = 0lu; i < shape_item.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(shape_item.second[i]);
            result << "}_";
        }
    }

    result << "FrameSize=" << frame_size << "_";
    result << "StepSize=" << frame_step << "_";
    result << "TransposeFrames=" << transpose_frames << "_";

    result << "ModelType=" << data_type << "_";
    result << "StepSizeType=" << step_size_type << "_";
    result << "IsParameterOnly=" << param_type << "_";

    result << "Device=" << dev;
    return result.str();
}

void STFTLayerTest::SetUp() {
    std::vector<InputShape> data_shapes;
    int64_t frame_size;          // frame size value
    int64_t frame_step;          // frame step value
    bool transpose_frames;       // transpose_frames
    ElementType data_type;       // data type
    ElementType step_size_type;  // size/step type
    utils::InputLayerType param_type;

    std::tie(data_shapes,
             frame_size,
             frame_step,
             transpose_frames,
             data_type,
             step_size_type,
             param_type,
             targetDevice) = this->GetParam();

    init_input_shapes(data_shapes);

    const auto in_signal = std::make_shared<ov::op::v0::Parameter>(data_type, inputDynamicShapes[0]);
    const auto in_window = std::make_shared<ov::op::v0::Parameter>(data_type, inputDynamicShapes[1]);

    if (param_type == utils::InputLayerType::PARAMETER) {
        const auto in_frame_size = std::make_shared<ov::op::v0::Parameter>(step_size_type, ov::Shape{});
        const auto in_frame_step = std::make_shared<ov::op::v0::Parameter>(step_size_type, ov::Shape{});
        const auto STFT =
            std::make_shared<ov::op::v15::STFT>(in_signal, in_window, in_frame_size, in_frame_step, transpose_frames);
        function = std::make_shared<ov::Model>(STFT->outputs(),
                                               ov::ParameterVector{in_signal, in_window, in_frame_size, in_frame_step});
    } else {
        const auto in_frame_size = std::make_shared<ov::op::v0::Constant>(step_size_type, ov::Shape{}, frame_size);
        const auto in_frame_step = std::make_shared<ov::op::v0::Constant>(step_size_type, ov::Shape{}, frame_step);
        const auto STFT =
            std::make_shared<ov::op::v15::STFT>(in_signal, in_window, in_frame_size, in_frame_step, transpose_frames);
        function = std::make_shared<ov::Model>(STFT->outputs(), ov::ParameterVector{in_signal, in_window});
    }
}
}  // namespace test
}  // namespace ov
