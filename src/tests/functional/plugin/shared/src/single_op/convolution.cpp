// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/convolution.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convolution.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
std::string ConvolutionLayerTest::getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj) {
    convSpecificParams conv_params;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::string targetDevice;
    std::tie(conv_params, model_type, shapes, targetDevice) = obj.param;
    ov::op::PadType pad_type;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> pad_begin, pad_end;
    size_t conv_out_channels;
    std::tie(kernel, stride, pad_begin, pad_end, dilation, conv_out_channels, pad_type) = conv_params;

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
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(pad_begin) << "_";
    result << "PE" << ov::test::utils::vec2str(pad_end) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << conv_out_channels << "_";
    result << "AP=" << pad_type << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ConvolutionLayerTest::SetUp() {
    convSpecificParams conv_params;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::tie(conv_params, model_type, shapes, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    ov::op::PadType pad_type;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> pad_begin, pad_end;
    size_t conv_out_channels;
    std::tie(kernel, stride, pad_begin, pad_end, dilation, conv_out_channels, pad_type) = conv_params;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front())};

    ov::Shape filterWeightsShape = {conv_out_channels, static_cast<size_t>(inputDynamicShapes.front()[1].get_length())};
    filterWeightsShape.insert(filterWeightsShape.end(), kernel.begin(), kernel.end());

    auto tensor = ov::test::utils::create_and_fill_tensor(model_type, filterWeightsShape);
    auto filter_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
    auto conv = std::make_shared<ov::op::v1::Convolution>(params[0], filter_weights_node, stride, pad_begin, pad_end, dilation, pad_type);

    function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(conv), params, "convolution");
}
}  // namespace test
}  // namespace ov
