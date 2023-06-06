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
    result << ";resize_algorithm=" << std::get<1>(obj.param);
    return result.str();
}

void PreprocessingResizeTests::SetUp() {
    const auto& test_params = this->GetParam();
    this->targetDevice = std::get<0>(test_params);

    this->function = build_test_model();
    PrePostProcessor ppp(this->function);
    ppp.input().tensor().set_shape({1, 1, -1, -1}).set_layout("NCHW");
    ppp.input(0).preprocess().resize(std::get<1>(test_params));
    ppp.build();

    this->inputs.insert({this->function->get_parameters().at(0), get_input_tensor()});
}

void PreprocessingResizeTests::run() {
    compile_model();
    // inference and output tensors retrieval
    validate();
}

std::vector<ov::Tensor> PreprocessingResizeTests::calculate_refs() {
    const auto& test_params = this->GetParam();
    const auto& values = std::get<2>(test_params);
    ov::Tensor out_tensor(element::f32, Shape{1, 1, 4, 4});
    auto* dst = out_tensor.data<float>();
    std::copy(values.begin(), values.end(), dst);

    return {out_tensor};
}

// static std::string getModelFullPath(const char* path) {
//     return FileUtils::makePath<char>(FileUtils::makePath<char>(CommonTestUtils::getExecutableDirectory(),
//     TEST_MODELS),
//                                      path);
// }

// void QuantizedModelsTests::runModel(const char* model, const LayerInputTypes& expected_layer_input_types, float thr)
// {
//     threshold = thr;
//     auto ie = getCore();
//     auto network = ie->ReadNetwork(getModelFullPath(model));
//     function = network.getFunction();
//     Run();
//     auto runtime_function = executableNetwork.GetExecGraphInfo().getFunction();
//     int ops_found = 0;
//     for (const auto& node : runtime_function->get_ordered_ops()) {
//         const auto& name = node->get_friendly_name();
//         if (expected_layer_input_types.count(name)) {
//             ops_found++;
//             const auto& expected_input_types = expected_layer_input_types.at(name);
//             auto inputs = node->input_values();
//             ASSERT_EQ(inputs.size(), expected_input_types.size());
//             for (size_t i = 0; i < inputs.size(); i++)
//                 ASSERT_EQ(expected_input_types[i], inputs[i].get_element_type());
//         }
//     }
//     ASSERT_GT(ops_found, 0);
// }

TEST_P(PreprocessingResizeTests, BilinearPillow) {
    // SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

}  // namespace preprocess
}  // namespace ov
