// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// DEPRECATED, can't be removed currently due to arm and kmb-plugin dependency (#55568)

#include "shared_test_classes/single_op/convolution_backprop_data.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convolution.hpp"
#include "common_test_utils/node_builders/convolution_backprop_data.hpp"

namespace ov {
namespace test {
std::string ConvolutionBackpropDataLayerTest::getTestCaseName(const testing::TestParamInfo<convBackpropDataLayerTestParamsSet>& obj) {
    convBackpropDataSpecificParams convBackpropDataParams;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    ov::Shape output_shapes;
    std::string target_device;
    std::tie(convBackpropDataParams, model_type, shapes, output_shapes, target_device) = obj.param;
    ov::op::PadType pad_type;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> pad_begin, pad_end, out_padding;
    size_t convOutChannels;
    std::tie(kernel, stride, pad_begin, pad_end, dilation, convOutChannels, pad_type, out_padding) = convBackpropDataParams;

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
    result << "OS=" << ov::test::utils::vec2str(output_shapes) << "_";
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(pad_begin) << "_";
    result << "PE" << ov::test::utils::vec2str(pad_end) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "OP=" << ov::test::utils::vec2str(out_padding) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << pad_type << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ConvolutionBackpropDataLayerTest::SetUp() {
    convBackpropDataSpecificParams convBackpropDataParams;
    std::vector<InputShape> shapes;
    ov::Shape output_shape;
    ov::element::Type model_type;
    std::tie(convBackpropDataParams, model_type, shapes, output_shape, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    ov::op::PadType pad_type;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> pad_begin, pad_end, out_padding;
    size_t convOutChannels;
    std::tie(kernel, stride, pad_begin, pad_end, dilation, convOutChannels, pad_type, out_padding) = convBackpropDataParams;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front())};

    std::shared_ptr<ov::Node> convBackpropData;
    if (!output_shape.empty()) {
        auto outShape = ov::op::v0::Constant::create(ov::element::i64, {output_shape.size()}, output_shape);
        convBackpropData = ov::test::utils::make_convolution_backprop_data(
            params[0]->output(0), outShape, model_type, kernel, stride, pad_begin, pad_end, dilation, pad_type, convOutChannels);
    } else {
        convBackpropData = ov::test::utils::make_convolution_backprop_data(
            params[0]->output(0), model_type, kernel, stride, pad_begin, pad_end, dilation, pad_type, convOutChannels, false, out_padding);
    }
    function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(convBackpropData), params, "convolutionBackpropData");
}
}  // namespace test
}  // namespace ov
