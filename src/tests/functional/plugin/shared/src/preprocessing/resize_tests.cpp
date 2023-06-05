// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include <file_utils.h>
// #include "common_test_utils/file_utils.hpp"
#include "preprocessing/resize_tests.hpp"

#include "openvino/op/util/attr_types.hpp"

namespace ov {

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
    targetDevice = std::get<0>(this->GetParam());
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
    // runModel("max_pool_qdq.onnx", {{"890_original", {ngraph::element::u8}}}, 1e-5);
}

}  // namespace preprocess
}  // namespace ov
