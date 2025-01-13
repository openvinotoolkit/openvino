// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/single_op/deformable_convolution.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/deformable_convolution.hpp"


namespace ov {
namespace test {
std::string DeformableConvolutionLayerTest::getTestCaseName(const testing::TestParamInfo<deformableConvLayerTestParamsSet>& obj) {
    deformableConvSpecificParams convParams;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::string target_device;
    bool with_modulation;
    std::tie(convParams, with_modulation, model_type, shapes, target_device) = obj.param;
    ov::op::PadType padType;
    std::vector<size_t> stride, dilation;
    std::vector<ptrdiff_t> pad_begin, pad_end;
    size_t groups, deformable_groups, conv_out_channels;
    bool with_bilinear_interpolation_pad;
    std::tie(stride, pad_begin, pad_end, dilation, groups, deformable_groups, conv_out_channels, padType, with_bilinear_interpolation_pad) = convParams;

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
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(pad_begin) << "_";
    result << "PE" << ov::test::utils::vec2str(pad_end) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "G=" << groups << "_";
    result << "DG=" << deformable_groups << "_";
    result << "O=" << conv_out_channels << "_";
    result << "AP=" << padType << "_";
    result << "BI_PAD=" << with_bilinear_interpolation_pad << "_";
    result << "MODULATION=" << with_modulation << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void DeformableConvolutionLayerTest::SetUp() {
    deformableConvSpecificParams convParams;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    bool with_modulation;
    std::tie(convParams, with_modulation, model_type, shapes, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    ov::op::PadType padType;
    std::vector<size_t> stride, dilation;
    std::vector<ptrdiff_t> pad_begin, pad_end;
    size_t groups, deformable_groups, conv_out_channels;
    bool with_bilinear_interpolation_pad;
    std::tie(stride, pad_begin, pad_end, dilation, groups, deformable_groups, conv_out_channels, padType, with_bilinear_interpolation_pad) = convParams;

    auto data = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]);
    data->set_friendly_name("a_data");
    auto offset_vals = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]);
    offset_vals->set_friendly_name("b_offset_vals");
    auto filter_vals = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2]);
    filter_vals->set_friendly_name("c_filter_vals");

    ov::ParameterVector parameters{data, offset_vals, filter_vals};
    std::shared_ptr<ov::Node> deformable_conv;
    if (with_modulation) {
        auto modulation_scalars = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[3]);
        modulation_scalars->set_friendly_name("c_modulation_scalars");

        deformable_conv = std::make_shared<ov::op::v8::DeformableConvolution>(data, offset_vals, filter_vals, modulation_scalars, stride, pad_begin,
                                                                                  pad_end, dilation, padType, groups, deformable_groups,
                                                                                  with_bilinear_interpolation_pad);
        parameters.push_back(modulation_scalars);
    } else {
        deformable_conv = std::make_shared<ov::op::v8::DeformableConvolution>(data, offset_vals, filter_vals, stride, pad_begin, pad_end, dilation,
                                                                                  padType, groups, deformable_groups, with_bilinear_interpolation_pad);
    }

    auto result = std::make_shared<ov::op::v0::Result>(deformable_conv);
    function = std::make_shared<ov::Model>(result, parameters, "deformable_convolution");
}
}  // namespace test
}  // namespace ov
