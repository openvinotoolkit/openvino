// Copyright (C) 2018-2025 Intel Corporation
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
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");
/*
// ############################################################################ CORE TESTS
OPENVINO_TEST(${BACKEND_NAME}, onnx_compress_axis_0) {
    ov::onnx_editor::ONNXModelEditor editor{
        util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "compress_0.onnx"})};

    std::map<std::string, std::shared_ptr<op::v0::Constant>> in_vals;

    in_vals.emplace("input", op::v0::Constant::create( ov::element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition", op::v0::Constant::create( ov::element::boolean, Shape{3}, {false, true, true}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{2, 2}, {3., 4., 5., 6.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_compress_axis_1) {
    ov::onnx_editor::ONNXModelEditor editor{
        util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "compress_1.onnx"})};

    std::map<std::string, std::shared_ptr<op::v0::Constant>> in_vals;

    in_vals.emplace("input", op::v0::Constant::create( ov::element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition", op::v0::Constant::create( ov::element::boolean, Shape{2}, {false, true}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{3, 1}, {2., 4., 6.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_compress_default_axis) {
    ov::onnx_editor::ONNXModelEditor editor{util::path_join(
        {ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "compress_default_axis.onnx"})};

    std::map<std::string, std::shared_ptr<op::v0::Constant>> in_vals;

    in_vals.emplace("input", op::v0::Constant::create( ov::element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition",
                    op::v0::Constant::create( ov::element::boolean, Shape{5}, {false, true, false, false, true}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{2}, {2., 5.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_compress_negative_axis) {
    ov::onnx_editor::ONNXModelEditor editor{util::path_join(
        {ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "compress_negative_axis.onnx"})};

    std::map<std::string, std::shared_ptr<op::v0::Constant>> in_vals;

    in_vals.emplace("input", op::v0::Constant::create( ov::element::f32, Shape{3, 2}, {1., 2., 3., 4., 5., 6.}));
    in_vals.emplace("condition", op::v0::Constant::create( ov::element::boolean, Shape{2}, {false, true}));
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
    const  ov::element::Type ng_type =  ov::element::from<DataType>();

    ov::onnx_editor::ONNXModelEditor editor{
        util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "add_abc_3d.onnx"})};

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
    const  ov::element::Type ng_type =  ov::element::from<DataType>();

    ov::onnx_editor::ONNXModelEditor editor{util::path_join(
        {ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "split_equal_parts_default.onnx"})};

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
*/
