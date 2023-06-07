// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include <file_utils.h>
// #include "common_test_utils/file_utils.hpp"
#include "preprocessing/resize_tests.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace {
std::shared_ptr<ov::Model> build_test_model() {
    // auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    // data1->set_friendly_name("input1");
    // data1->get_output_tensor(0).set_names({"tensor_input1"});
    // auto op = std::make_shared<op::v0::Relu>(data1);
    // op->set_friendly_name("Relu");
    // op->get_output_tensor(0).set_names({"tensor_Relu"});
    // auto res = std::make_shared<op::v0::Result>(op);
    // res->set_friendly_name("Result1");
    // res->get_output_tensor(0).set_names({"tensor_output1"});
    // return std::make_shared<Model>(ResultVector{res}, ParameterVector{data1});
    const auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, 4, 4});
    const auto zero = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    const auto op = std::make_shared<op::v1::Add>(input, zero);
    const auto res = std::make_shared<op::v0::Result>(op);
    return std::make_shared<ov::Model>(res, ParameterVector{input});
}

ov::Tensor get_input_tensor() {
    ov::Tensor input_tensor(element::f32, Shape{1, 1, 2, 2});
    const auto input_values = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto* dst = input_tensor.data<float>();
    std::copy(input_values.begin(), input_values.end(), dst);
    return input_tensor;
}
}  // namespace

namespace preprocess {
std::ostream& operator<<(std::ostream& s, const ResizeAlgorithm& algo) {
    static std::map<preprocess::ResizeAlgorithm, std::string> enum_names = {
        {preprocess::ResizeAlgorithm::RESIZE_LINEAR, "RESIZE_LINEAR"},
        {preprocess::ResizeAlgorithm::RESIZE_CUBIC, "RESIZE_CUBIC"},
        {preprocess::ResizeAlgorithm::RESIZE_NEAREST, "RESIZE_NEAREST"},
        {preprocess::ResizeAlgorithm::RESIZE_BILINEAR_PILLOW, "RESIZE_BILINEAR_PILLOW"},
        {preprocess::ResizeAlgorithm::RESIZE_BICUBIC_PILLOW, "RESIZE_BICUBIC_PILLOW"}};

    return s << enum_names[algo];
}

std::string PreprocessingResizeTests::getTestCaseName(const testing::TestParamInfo<ResizeTestsParams>& obj) {
    std::ostringstream result;
    result << "device=" << std::get<0>(obj.param);
    return result.str();
}

void PreprocessingResizeTests::SetUp() {
    const auto& test_params = this->GetParam();
    this->targetDevice = std::get<0>(test_params);

    this->function = build_test_model();
}

void PreprocessingResizeTests::run() {
    compile_model();
    // inference, output tensors retrieval and tensors comparison:
    validate();
}

void PreprocessingResizeTests::run_with_algorithm(const ResizeAlgorithm algo,
                                                  const std::vector<float>& expected_output) {
    PrePostProcessor ppp(this->function);
    ppp.input().tensor().set_shape({1, 1, -1, -1}).set_layout("NCHW");
    ppp.input(0).preprocess().resize(algo);
    ppp.build();

    this->inputs.insert({this->function->get_parameters().at(0), get_input_tensor()});

    ov::Tensor out_tensor(element::f32, Shape{1, 1, 4, 4});
    auto* dst = out_tensor.data<float>();
    std::copy(expected_output.begin(), expected_output.end(), dst);
    expected_output_tensor = out_tensor;

    run();
}

std::vector<ov::Tensor> PreprocessingResizeTests::calculate_refs() {
    return {expected_output_tensor};
}

TEST_P(PreprocessingResizeTests, BilinearPillow) {
    // SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto expected_output =
        std::vector<float>{1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0};
    run_with_algorithm(ResizeAlgorithm::RESIZE_LINEAR, expected_output);
}

}  // namespace preprocess
}  // namespace ov
