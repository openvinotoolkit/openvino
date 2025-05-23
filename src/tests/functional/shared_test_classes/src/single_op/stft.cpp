// Copyright (C) 2018-2025 Intel Corporation
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

const STFTLayerTest::TGenData STFTLayerTest::GetTestDataForDevice(const char* deviceName) {
    const std::vector<ov::element::Type> data_type = {ov::element::bf16, ov::element::f16};
    const std::vector<ov::element::Type> step_size_type = {ov::element::i32, ov::element::i64};

    const std::vector<std::vector<InputShape>> input_shapes = {
        {
            // Static shapes
            {{}, {{128}}},  // 1st input
            {{}, {{8}}},    // 2nd input
            {{}, {{}}},     // 3rd input
            {{}, {{}}}      // 4th input
        },
        {
            // Static shapes
            {{}, {{1, 128}}},  // 1st input
            {{}, {{8}}},       // 2nd input
            {{}, {{}}},        // 3rd input
            {{}, {{}}}         // 4th input
        },
        {
            // Static shapes
            {{}, {{2, 226}}},  // 1st input
            {{}, {{16}}},      // 2nd input
            {{}, {{}}},        // 3rd input
            {{}, {{}}}         // 4th input
        },
        {
            // Dynamic dims in the first input shape
            {{-1, -1}, {{1, 128}, {2, 226}}},  // 1st input
            {{}, {{8}}},                       // 2nd input
            {{}, {{}}},                        // 3rd input
            {{}, {{}}}                         // 4th input
        },
        {
            // Dynamic dims in the first and second input shape
            {{-1}, {{128}}},  // 1st input
            {{-1}, {{8}}},    // 2nd input
            {{}, {{}}},       // 3rd input
            {{}, {{}}}        // 4th input
        },
        {
            // Dynamic dims in the first and second input shape
            {{-1, -1}, {{1, 128}, {2, 226}}},  // 1st input
            {{-1}, {{8}, {16}}},               // 2nd input
            {{}, {{}}},                        // 3rd input
            {{}, {{}}}                         // 4th input
        },
        {
            // Dynamic dims with range in the first and second input shape
            {{{2, 4}, {1, 300}}, {{2, 226}, {3, 128}}},  // 1st input
            {{{3, 16}}, {{4}, {16}}},                    // 2nd input
            {{}, {{}}},                                  // 3rd input
            {{}, {{}}}                                   // 4th input
        }};

    const std::vector<int64_t> frame_size = {16, 24};
    const std::vector<int64_t> step_size = {2, 3, 4};

    const std::vector<bool> transpose_frames = {
        false,
        true,
    };

    std::vector<utils::InputLayerType> in_types = {utils::InputLayerType::CONSTANT, utils::InputLayerType::PARAMETER};

    auto data = ::testing::Combine(::testing::ValuesIn(input_shapes),
                                   ::testing::ValuesIn(frame_size),
                                   ::testing::ValuesIn(step_size),
                                   ::testing::ValuesIn(transpose_frames),
                                   ::testing::ValuesIn(data_type),
                                   ::testing::ValuesIn(step_size_type),
                                   ::testing::ValuesIn(in_types),
                                   ::testing::Values(deviceName));

    return data;
}
}  // namespace test
}  // namespace ov
