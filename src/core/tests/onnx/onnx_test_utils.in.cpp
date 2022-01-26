// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <sstream>

#include "default_opset.hpp"
#include "editor.hpp"
#include "engines_util/test_case.hpp"
#include "engines_util/test_engines.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "onnx_import/onnx.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
OPENVINO_SUPPRESS_DEPRECATED_START

static std::string s_manifest = "${MANIFEST}";
static std::string s_device = test::backend_name_to_device("${BACKEND_NAME}");

template <typename T>
class ElemTypesTests : public ::testing::Test {};
TYPED_TEST_SUITE_P(ElemTypesTests);

TYPED_TEST_P(ElemTypesTests, onnx_test_add_abc_set_precission) {
    using DataType = TypeParam;
    const element::Type ng_type = element::from<DataType>();

    ov::onnx_editor::ONNXModelEditor editor{file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_3d.onnx")};

    editor.set_input_types({{"A", ng_type}, {"B", ng_type}, {"C", ng_type}});

    const auto function = editor.get_function();
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<DataType>(std::vector<DataType>{1, 2, 3});
    test_case.add_input<DataType>(std::vector<DataType>{4, 5, 6});
    test_case.add_input<DataType>(std::vector<DataType>{7, 8, 9});
    test_case.add_expected_output<DataType>(Shape{3}, std::vector<DataType>{12, 15, 18});
    test_case.run();
}

TYPED_TEST_P(ElemTypesTests, onnx_test_split_multioutput_set_precission) {
    using DataType = TypeParam;
    const element::Type ng_type = element::from<DataType>();

    ov::onnx_editor::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/split_equal_parts_default.onnx")};

    editor.set_input_types({{"input", ng_type}});

    const auto function = editor.get_function();
    auto test_case = test::TestCase(function, s_device);
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

NGRAPH_TEST(${BACKEND_NAME}, add_abc_from_ir) {
    const auto ir_xml = file_util::path_join(SERIALIZED_ZOO, "ir/add_abc.xml");
    const auto function = test::function_from_ir(ir_xml);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1});
    test_case.add_input<float>({2});
    test_case.add_input<float>({3});
    test_case.add_expected_output<float>(Shape{1}, {6});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, add_abc_from_ir_with_bin_path) {
    const auto ir_xml = file_util::path_join(SERIALIZED_ZOO, "ir/add_abc.xml");
    const auto ir_bin = file_util::path_join(SERIALIZED_ZOO, "ir/weights/add_abc.bin");
    const auto function = test::function_from_ir(ir_xml, ir_bin);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1});
    test_case.add_input<float>({2});
    test_case.add_input<float>({3});
    test_case.add_expected_output<float>(Shape{1}, {6});

    test_case.run();
}
