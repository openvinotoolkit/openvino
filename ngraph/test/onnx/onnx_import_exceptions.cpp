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

#include <exception>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

TEST(onnx_importer, exception_throws_ngraph_error)
{
    EXPECT_THROW(onnx_import::import_onnx_model(file_util::path_join(
                     SERIALIZED_ZOO, "onnx/depth_to_space_bad_blocksize.prototxt")),
                 ngraph_error);
}

TEST(onnx_importer, exception_msg_ngraph_error)
{
    try
    {
        onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/depth_to_space_bad_blocksize.prototxt"));
        // Should have thrown, so fail if it didn't
        FAIL() << "ONNX Importer did not detected incorrect model!";
    }
    catch (const ngraph_error& e)
    {
        EXPECT_HAS_SUBSTRING(e.what(),
                             std::string("While validating ONNX node '<Node(DepthToSpace)"));
        EXPECT_HAS_SUBSTRING(e.what(), std::string("While validating node 'v0::DepthToSpace"));
    }
    catch (...)
    {
        FAIL() << "The ONNX model importer failed for unexpected reason";
    }
}

TEST(onnx_importer, exception_msg_onnx_node_validation_failure)
{
    try
    {
        onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/instance_norm_bad_scale_type.prototxt"));
        // Should have thrown, so fail if it didn't
        FAIL() << "ONNX Importer did not detected incorrect model!";
    }
    catch (const onnx_import::error::OnnxNodeValidationFailure& e)
    {
        EXPECT_HAS_SUBSTRING(
            e.what(), std::string("While validating ONNX node '<Node(InstanceNormalization)"));
    }
    catch (...)
    {
        FAIL() << "The ONNX model importer failed for unexpected reason";
    }
}

// This test aims to check for wrapping all std::exception not deriving from ngraph_error.
// This test should throw a std error because of attempt to access shape from dynamic tensor.
TEST(onnx_importer, exception_msg_std_err_wrapped)
{
    try
    {
        onnx_import::import_onnx_model(file_util::path_join(
            SERIALIZED_ZOO, "onnx/dynamic_shapes/add_opset6_dyn_shape.prototxt"));
        // Should have thrown, so fail if it didn't
        FAIL() << "ONNX Importer did not detected incorrect model!";
    }
    catch (const std::exception& e)
    {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("While validating ONNX node '<Node(Add)"));
    }
    catch (...)
    {
        FAIL() << "The ONNX model importer failed for unexpected reason";
    }
}
