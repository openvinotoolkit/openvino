// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"

#include "shared_test_classes/single_op/col2im.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

using ov::test::InputShape;
using ov::test::Col2Im::Col2ImOpsSpecificParams;
using ov::test::Col2Im::Col2ImLayerSharedTestParams;

namespace ov {
namespace test {
namespace Col2Im {

std::string Col2ImLayerSharedTest::getTestCaseName(const testing::TestParamInfo<Col2ImLayerSharedTestParams>& obj) {
    ov::element::Type data_precision;
    ov::element::Type size_precision;
    std::string target_device;
    Col2ImOpsSpecificParams col2im_param;

    std::tie(col2im_param, data_precision, size_precision, target_device) = obj.param;

    ov::Shape input_shape;
    std::vector<int64_t> output_size;
    std::vector<int64_t> kernel_size;
    ov::Strides strides;
    ov::Strides dilations;
    ov::Shape pads_begin;
    ov::Shape pads_end;
    std::tie(input_shape, output_size, kernel_size, strides, dilations, pads_begin, pads_end) = col2im_param;

    std::ostringstream result;

    result << data_precision << "_IS=";
    result << input_shape << "_";
    result << "TS=";
    result << "(";
    result << input_shape;
    result << ")_";
    result << "output_size=" << ov::test::utils::vec2str(output_size) << "_";
    result << "kernel_size=" << ov::test::utils::vec2str(kernel_size) << "_";
    result << "strides=" << strides << "_";
    result << "dilations=" << dilations << "_";
    result << "padsBegin=" << pads_begin << "_";
    result << "padsEnd=" << pads_end << "_";
    result << "dataPrecision=" << data_precision << "_";
    result << "constSizePrecision=" << size_precision << "_";
    result << "trgDev=" << target_device;

    return result.str();
}

void Col2ImLayerSharedTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();

    ov::Tensor data_tensor;
    const auto& dataPrecision = funcInputs[0].get_element_type();
    const auto& dataShape = targetInputStaticShapes.front();
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = 10;
    in_data.resolution = 1000;
    data_tensor = ov::test::utils::create_and_fill_tensor(dataPrecision, dataShape, in_data);
    inputs.insert({ funcInputs[0].get_node_shared_ptr(), data_tensor });
}

void Col2ImLayerSharedTest::SetUp() {
    Col2ImOpsSpecificParams col2im_param;
    ov::element::Type data_precision;
    ov::element::Type size_precision;
    std::string target_device;

    std::tie(col2im_param, data_precision, size_precision, target_device) = this->GetParam();

    targetDevice = target_device;

    ov::Shape data_input_shape;
    std::vector<int64_t> outputSize;
    std::vector<int64_t> kernelSize;
    ov::Strides strides;
    ov::Strides dilations;
    ov::Shape pads_begin;
    ov::Shape pads_end;
    std::tie(data_input_shape, outputSize, kernelSize, strides, dilations, pads_begin, pads_end) = col2im_param;

    auto dataParameter = std::make_shared<ov::op::v0::Parameter>(data_precision, data_input_shape);
    // Required option which is fixed size input : ov::Shape{2}
    auto outputSizeConst = std::make_shared<ov::op::v0::Constant>(size_precision, ov::Shape{ 2 }, outputSize);
    auto kernelSizeConst = std::make_shared<ov::op::v0::Constant>(size_precision, ov::Shape{ 2 }, kernelSize);

    auto Col2Im_node = std::make_shared<ov::op::v15::Col2Im>(dataParameter,
                                                        outputSizeConst,
                                                        kernelSizeConst,
                                                        strides,
                                                        dilations,
                                                        pads_begin,
                                                        pads_end);

    ov::ResultVector results;
    for (size_t i = 0; i < Col2Im_node->get_output_size(); i++)
            results.push_back(std::make_shared<ov::op::v0::Result>(Col2Im_node->output(i)));

    ov::ParameterVector params{ dataParameter };
    function = std::make_shared<ov::Model>(results, params, "col2im");

    if (data_precision == ov::element::f16) {
        rel_threshold = 0.01;
    }
}

} // namespace Col2Im
} // namespace test
} // namespace ov
