//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/onnx.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

namespace
{
    using InputTypePred = std::function<bool(const std::shared_ptr<ngraph::Node>)>;

    InputTypePred element_type_is(const element::Type et)
    {
        return [et](const std::shared_ptr<ngraph::Node> input) {
            return std::dynamic_pointer_cast<onnx_import::default_opset::Parameter>(input)
                       ->get_element_type() == et;
        };
    }
} // namespace

NGRAPH_TEST(${BACKEND_NAME}, onnx_editor_single_input_type_substitution)
{
    // the original model contains 3 inputs with f32 data type
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.prototxt"));

    const auto all_ops_in_graph = function->get_ops();
    std::vector<std::shared_ptr<ngraph::Node>> graph_inputs;
    std::copy_if(std::begin(all_ops_in_graph),
                 std::end(all_ops_in_graph),
                 std::back_inserter(graph_inputs),
                 [](const std::shared_ptr<ngraph::Node> node) { return op::is_parameter(node); });

    const auto float_inputs_count = std::count_if(
        std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::f32));

    const auto integer_inputs_count = std::count_if(
        std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::i32));

    EXPECT_EQ(float_inputs_count, 2);
    EXPECT_EQ(integer_inputs_count, 1);
}
