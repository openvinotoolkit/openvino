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

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/onnx.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, onnx_editor_single_input_type_substitution)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.prototxt"));

    unsigned int float_inputs = 0;
    unsigned int integer_inputs = 0;
    for (const auto& op : function->get_ops())
    {
        if (const auto param = std::dynamic_pointer_cast<onnx_import::default_opset::Parameter>(op))
        {
            if (param->get_element_type() == element::f32)
            {
                ++float_inputs;
                continue;
            }
        }
    }

    EXPECT_EQ(float_inputs, 1);
    EXPECT_EQ(integer_inputs, 1);
}
