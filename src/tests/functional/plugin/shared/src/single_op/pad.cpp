// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/pad.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/pad.hpp"

namespace ov {
namespace test {
std::string PadLayerTest::getTestCaseName(const testing::TestParamInfo<padLayerTestParamsSet>& obj) {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::vector<int64_t> pads_begin, pads_end;
    ov::op::PadMode pad_mode;
    float arg_pad_value;
    std::string target_device;
    std::tie(pads_begin, pads_end, arg_pad_value, pad_mode, model_type, shapes, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "PadsBegin=" << ov::test::utils::vec2str(pads_begin) << "_";
    result << "PadsEnd=" << ov::test::utils::vec2str(pads_end) << "_";
    if (pad_mode == ov::op::PadMode::CONSTANT) {
        result << "Value=" << arg_pad_value << "_";
    }
    result << "PadMode=" << pad_mode << "_";
    result << "ModelType=" << model_type.get_type_name() << "_";
    result << "TrgDev=" << target_device;
    return result.str();
}

void PadLayerTest::SetUp() {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::vector<int64_t> pads_begin, pads_end;
    ov::op::PadMode pad_mode;
    float arg_pad_value;
    std::tie(pads_begin, pads_end, arg_pad_value, pad_mode, model_type, shapes, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto pads_begin_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{pads_begin.size()}, pads_begin.data());
    auto pads_end_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{pads_end.size()}, pads_end.data());
    auto arg_pad_value_const = std::make_shared<ov::op::v0::Constant>(model_type, ov::Shape{}, &arg_pad_value);

    auto pad = create_pad_op(param, pads_begin_const, pads_end_const, arg_pad_value_const, pad_mode);

    auto result = std::make_shared<ov::op::v0::Result>(pad);

    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "pad");
}

std::shared_ptr<ov::Node> PadLayerTest::create_pad_op(const std::shared_ptr<ov::Node>& data,
                                        const std::shared_ptr<ov::Node>& pads_begin,
                                        const std::shared_ptr<ov::Node>& pads_end,
                                        const std::shared_ptr<ov::Node>& arg_pad_value,
                                        ov::op::PadMode pad_mode) const {
    return std::make_shared<ov::op::v1::Pad>(data, pads_begin, pads_end, arg_pad_value, pad_mode);
}

std::shared_ptr<ov::Node> Pad12LayerTest::create_pad_op(const std::shared_ptr<ov::Node>& data,
                                        const std::shared_ptr<ov::Node>& pads_begin,
                                        const std::shared_ptr<ov::Node>& pads_end,
                                        const std::shared_ptr<ov::Node>& arg_pad_value,
                                        ov::op::PadMode pad_mode) const {
    return std::make_shared<ov::op::v12::Pad>(data, pads_begin, pads_end, arg_pad_value, pad_mode);
}
}  // namespace test
}  // namespace ov
