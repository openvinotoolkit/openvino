// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/istft.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"

namespace ov {
namespace test {

std::string ISTFTLayerTest::getTestCaseName(const testing::TestParamInfo<ISTFTParams>& obj) {
    std::ostringstream result;
    const std::vector<InputShape>& data_shapes = std::get<0>(obj.param);
    const int64_t frame_size = std::get<1>(obj.param);
    const int64_t frame_step = std::get<2>(obj.param);
    const int64_t signal_len = std::get<3>(obj.param);
    const bool center = std::get<4>(obj.param);
    const bool normalized = std::get<5>(obj.param);
    const ElementType& data_type = std::get<6>(obj.param);
    const ElementType& step_size_type = std::get<7>(obj.param);
    const utils::InputLayerType& param_type = std::get<8>(obj.param);
    const ov::test::TargetDevice& dev = std::get<9>(obj.param);

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
    result << "SignalLen=" << signal_len << "_";
    result << "Center=" << center << "_";
    result << "Normalized=" << normalized << "_";
    result << "ModelType=" << data_type << "_";
    result << "StepSizeType=" << step_size_type << "_";
    result << "IsParameterOnly=" << param_type << "_";

    result << "Device=" << dev;
    return result.str();
}

void ISTFTLayerTest::SetUp() {
    std::vector<InputShape> data_shapes;
    int64_t frame_size;          // frame size value
    int64_t frame_step;          // frame step value
    int64_t signal_len;          // signal length value
    bool center;                 // center
    bool normalized;             // normalized
    ElementType data_type;       // data_type
    ElementType step_size_type;  // size/step type
    utils::InputLayerType param_type;

    std::tie(data_shapes,
             frame_size,
             frame_step,
             signal_len,
             center,
             normalized,
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

        if (signal_len < 0) {
            const auto ISTFT = std::make_shared<ov::op::v16::ISTFT>(in_signal,
                                                                    in_window,
                                                                    in_frame_size,
                                                                    in_frame_step,
                                                                    center,
                                                                    normalized);
            function =
                std::make_shared<ov::Model>(ISTFT->outputs(),
                                            ov::ParameterVector{in_signal, in_window, in_frame_size, in_frame_step});

        } else {
            const auto signal_length = std::make_shared<ov::op::v0::Parameter>(step_size_type, ov::Shape{});

            const auto ISTFT = std::make_shared<ov::op::v16::ISTFT>(in_signal,
                                                                    in_window,
                                                                    in_frame_size,
                                                                    in_frame_step,
                                                                    signal_length,
                                                                    center,
                                                                    normalized);
            function = std::make_shared<ov::Model>(
                ISTFT->outputs(),
                ov::ParameterVector{in_signal, in_window, in_frame_size, in_frame_step, signal_length});
        }

    } else {
        const auto in_frame_size = std::make_shared<ov::op::v0::Constant>(step_size_type, ov::Shape{}, frame_size);
        const auto in_frame_step = std::make_shared<ov::op::v0::Constant>(step_size_type, ov::Shape{}, frame_step);

        if (signal_len < 0) {
            const auto ISTFT = std::make_shared<ov::op::v16::ISTFT>(in_signal,
                                                                    in_window,
                                                                    in_frame_size,
                                                                    in_frame_step,
                                                                    center,
                                                                    normalized);
            function = std::make_shared<ov::Model>(ISTFT->outputs(), ov::ParameterVector{in_signal, in_window});
        } else {
            const auto signal_length = std::make_shared<ov::op::v0::Constant>(step_size_type, ov::Shape{}, frame_step);
            const auto ISTFT = std::make_shared<ov::op::v16::ISTFT>(in_signal,
                                                                    in_window,
                                                                    in_frame_size,
                                                                    in_frame_step,
                                                                    signal_length,
                                                                    center,
                                                                    normalized);
            function = std::make_shared<ov::Model>(ISTFT->outputs(), ov::ParameterVector{in_signal, in_window});
        }
    }
}

const ISTFTLayerTest::TGenData ISTFTLayerTest::GetTestDataForDevice(const char* deviceName) {
    const std::vector<ov::element::Type> data_type = {ov::element::bf16, ov::element::f16};
    const std::vector<ov::element::Type> step_size_type = {ov::element::i32, ov::element::i64};

    const std::vector<std::vector<InputShape>> input_shapes = {
        {
            // Static shapes
            {{}, {{9, 3, 2}}},  // 1st input
            {{}, {{8}}},        // 2nd input
            {{}, {{}}},         // 3rd input
            {{}, {{}}},         // 4th input
            {{}, {{}}}          // 5th input
        },
        {
            // Dynamic dims in the first and second input shape
            {{-1, -1, -1}, {{9, 3, 2}}},  // 1st input
            {{-1}, {{8}}},                // 2nd input
            {{}, {{}}},                   // 3rd input
            {{}, {{}}},                   // 4th input
            {{}, {{}}}                    // 5th input
        }};

    const std::vector<int64_t> frame_size = {16, 24};
    const std::vector<int64_t> step_size = {2, 3, 4};
    const std::vector<int64_t> signal_len = {-1, 48, 32, 256};

    const std::vector<bool> center = {
        false,
        true,
    };

    const std::vector<bool> normalized = {
        false,
        true,
    };

    std::vector<utils::InputLayerType> in_types = {utils::InputLayerType::CONSTANT, utils::InputLayerType::PARAMETER};

    auto data = ::testing::Combine(::testing::ValuesIn(input_shapes),
                                   ::testing::ValuesIn(frame_size),
                                   ::testing::ValuesIn(step_size),
                                   ::testing::ValuesIn(signal_len),
                                   ::testing::ValuesIn(center),
                                   ::testing::ValuesIn(normalized),
                                   ::testing::ValuesIn(data_type),
                                   ::testing::ValuesIn(step_size_type),
                                   ::testing::ValuesIn(in_types),
                                   ::testing::Values(deviceName));

    return data;
}
}  // namespace test
}  // namespace ov
