// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/pad.hpp"

#include "ov_models/builders.hpp"
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
    ov::test::utils::PadMode pad_mode;
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
    if (pad_mode == ov::test::utils::PadMode::CONSTANT) {
        result << "Value=" << arg_pad_value << "_";
    }
    result << "PadMode=" << pad_mode << "_";
    result << "ModelType=" << model_type.get_type_name() << "_";
    result << "TrgDev=" << target_device;
    return result.str();
}

std::shared_ptr<ov::Node> PadLayerTest::create_pad_op(const ngraph::Output<ov::Node>& data,
                                    const std::vector<int64_t>& pads_begin,
                                    const std::vector<int64_t>& pads_end,
                                    float arg_pad_value,
                                    ov::test::utils::PadMode pad_mode) const {
    return ngraph::builder::makePad(data, pads_begin, pads_end, arg_pad_value, pad_mode, false);
}

void PadLayerTest::SetUp() {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::vector<int64_t> pads_begin, pads_end;
    ov::test::utils::PadMode pad_mode;
    float arg_pad_value;
    std::tie(pads_begin, pads_end, arg_pad_value, pad_mode, model_type, shapes, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto pad = create_pad_op(param, pads_begin, pads_end, arg_pad_value, pad_mode);

    auto result = std::make_shared<ngraph::opset3::Result>(pad);

    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "pad");
}

std::shared_ptr<ov::Node> PadLayerTest12::create_pad_op(const ngraph::Output<ov::Node>& data,
                                    const std::vector<int64_t>& pads_begin,
                                    const std::vector<int64_t>& pads_end,
                                    float arg_pad_value,
                                    ov::test::utils::PadMode pad_mode) const {
    return ngraph::builder::makePad(data, pads_begin, pads_end, arg_pad_value, pad_mode, true);
}
}  // namespace test
}  // namespace ov
