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

#include <memory>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/frontend/onnx_import/default_opset.hpp"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/all_close.hpp"
#include "util/test_control.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;
using namespace ngraph::onnx_import;

static std::string s_manifest = "${MANIFEST}";

namespace
{
    template <typename T>
    void test_constant_folding(std::shared_ptr<ngraph::Function> ng_function,
                               const std::vector<T> expected_output,
                               const PartialShape expected_shape = PartialShape::dynamic())
    {
        ngraph::pass::Manager pass_manager;
        pass_manager.register_pass<pass::ConstantFolding>();
        pass_manager.run_passes(ng_function);

        for (auto ng_node : ng_function->get_ordered_ops())
        {
            if (ng_node->is_constant())
            {
                const auto folded_node = as_type_ptr<default_opset::Constant>(ng_node);
                const auto output_values = folded_node->cast_vector<T>();

                EXPECT_TRUE(ngraph::test::all_close(expected_output, output_values));

                if (expected_shape.is_static())
                {
                    EXPECT_EQ(folded_node->get_output_shape(0), expected_shape.to_shape());
                }

                return;
            }
        }

        FAIL() << "ONNX model import with constant folding failed.";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_scatter_elements)
{
    const auto fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/scatter_elements_opset11.prototxt"));

    test_constant_folding<float>(fn, {1.0, 1.1, 3.0, 2.1, 5.0}, Shape{1, 5});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_scalar)
{
    const auto fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/non_zero_scalar.prototxt"));

    test_constant_folding<int64_t>(fn, {0}, Shape{1, 1});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_1d)
{
    const auto fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/non_zero_1d.prototxt"));

    test_constant_folding<int64_t>(fn, {1, 2, 4}, Shape{1, 3});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_1d_float)
{
    const auto fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/non_zero_1d_float.prototxt"));

    test_constant_folding<int64_t>(fn, {0, 1, 3, 4, 5, 6, 7, 8, 9});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_3d)
{
    const auto fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/non_zero_3d.prototxt"));

    // Vertical slices are 3D indices of non-zero elements in the input tensor
    // {0, 0, 0, 1, 1, 2, 2}
    // {0, 0, 1, 0, 1, 0, 1}
    // {0, 1, 1, 1, 0, 0, 1}
    test_constant_folding<int64_t>(fn,
                                   {0, 0, 0, 1, 1, 2, 2, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1});
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_const_folding_model_non_zero_2d_bool)
{
    const auto fn = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/non_zero_2d_bool.prototxt"));

    test_constant_folding<int64_t>(fn, {0, 1, 1, 0});
}
