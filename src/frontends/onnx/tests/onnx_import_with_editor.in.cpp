// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "common_test_utils/file_utils.hpp"
#include "editor.hpp"
#include "common_test_utils/test_case.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/file_util.hpp"
#include "common_test_utils/test_control.hpp"
#include "onnx_utils.hpp"

using namespace ngraph;
OPENVINO_SUPPRESS_DEPRECATED_START

static std::string s_manifest = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(), "${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

// ############################################################################ CORE TESTS
OPENVINO_TEST(${BACKEND_NAME}, onnx_compress_axis_0) {
    ov::onnx_editor::ONNXModelEditor editor{
        file_util::path_join(ov::test::utils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/compress_0.onnx")};

    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("input", op::Constant::create(element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition", op::Constant::create(element::boolean, Shape{3}, {false, true, true}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{2, 2}, {3., 4., 5., 6.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_compress_axis_1) {
    ov::onnx_editor::ONNXModelEditor editor{
        file_util::path_join(ov::test::utils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/compress_1.onnx")};

    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("input", op::Constant::create(element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition", op::Constant::create(element::boolean, Shape{2}, {false, true}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{3, 1}, {2., 4., 6.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_compress_default_axis) {
    ov::onnx_editor::ONNXModelEditor editor{file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                                 SERIALIZED_ZOO,
                                                                 "onnx/compress_default_axis.onnx")};

    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("input", op::Constant::create(element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition", op::Constant::create(element::boolean, Shape{5}, {false, true, false, false, true}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{2}, {2., 5.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_compress_negative_axis) {
    ov::onnx_editor::ONNXModelEditor editor{file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                                 SERIALIZED_ZOO,
                                                                 "onnx/compress_negative_axis.onnx")};

    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("input", op::Constant::create(element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition", op::Constant::create(element::boolean, Shape{2}, {false, true}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{3, 1}, {2., 4., 6.});
    test_case.run();
}

template <typename T>
class ElemTypesTests : public ::testing::Test {};
TYPED_TEST_SUITE_P(ElemTypesTests);

TYPED_TEST_P(ElemTypesTests, onnx_test_add_abc_set_precission) {
    using DataType = TypeParam;
    const element::Type ng_type = element::from<DataType>();

    ov::onnx_editor::ONNXModelEditor editor{
        file_util::path_join(ov::test::utils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/add_abc_3d.onnx")};

    editor.set_input_types({{"A", ng_type}, {"B", ng_type}, {"C", ng_type}});

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function, s_device);
    test_case.add_input<DataType>(std::vector<DataType>{1, 2, 3});
    test_case.add_input<DataType>(std::vector<DataType>{4, 5, 6});
    test_case.add_input<DataType>(std::vector<DataType>{7, 8, 9});
    test_case.add_expected_output<DataType>(Shape{3}, std::vector<DataType>{12, 15, 18});
    test_case.run();
}

TYPED_TEST_P(ElemTypesTests, onnx_test_split_multioutput_set_precission) {
    using DataType = TypeParam;
    const element::Type ng_type = element::from<DataType>();

    ov::onnx_editor::ONNXModelEditor editor{file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                                 SERIALIZED_ZOO,
                                                                 "onnx/split_equal_parts_default.onnx")};

    editor.set_input_types({{"input", ng_type}});

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function, s_device);
    test_case.add_input<DataType>(std::vector<DataType>{1, 2, 3, 4, 5, 6});
    test_case.add_expected_output<DataType>(Shape{2}, std::vector<DataType>{1, 2});
    test_case.add_expected_output<DataType>(Shape{2}, std::vector<DataType>{3, 4});
    test_case.add_expected_output<DataType>(Shape{2}, std::vector<DataType>{5, 6});
    test_case.run();
}

REGISTER_TYPED_TEST_SUITE_P(ElemTypesTests,
                            onnx_test_add_abc_set_precission,
                            onnx_test_split_multioutput_set_precission);
typedef ::testing::Types<int8_t, int16_t, int32_t, uint8_t, float> ElemTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(${BACKEND_NAME}, ElemTypesTests, ElemTypes);
