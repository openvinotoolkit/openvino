// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/extract_image_patches.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/extractimagepatches.hpp"


namespace ov {
namespace test {

std::string ExtractImagePatchesTest::getTestCaseName(const testing::TestParamInfo<extractImagePatchesTuple> &obj) {
    std::vector<InputShape> shapes;
    std::vector<size_t> kernel, strides, rates;
    ov::op::PadType pad_type;
    ov::element::Type model_type;
    std::string device_name;
    std::tie(shapes, kernel, strides, rates, pad_type, model_type, device_name) = obj.param;
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
    result << "model_type=" << model_type.get_type_name() << "_";
    result << "K=" << ov::test::utils::vec2str(kernel) << "_";
    result << "S=" << ov::test::utils::vec2str(strides) << "_";
    result << "R=" << ov::test::utils::vec2str(rates) << "_";
    result << "P=" << pad_type << "_";
    result << "trgDev=" << device_name;
    return result.str();
}

void ExtractImagePatchesTest::SetUp() {
    std::vector<InputShape> shapes;
    std::vector<size_t> kernel, strides, rates;
    ov::op::PadType pad_type;
    ov::element::Type model_type;
    std::tie(shapes, kernel, strides, rates, pad_type, model_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto extImgPatches = std::make_shared<ov::op::v3::ExtractImagePatches>(param, ov::Shape(kernel), ov::Strides(strides), ov::Shape(rates), pad_type);
    auto result = std::make_shared<ov::op::v0::Result>(extImgPatches);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "ExtractImagePatches");
}
}  // namespace test
}  // namespace ov
