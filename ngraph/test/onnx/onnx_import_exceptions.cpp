// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <exception>

#include "exceptions.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "onnx_import/onnx.hpp"
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
    catch (const ::ngraph::onnx_import::error::OnnxNodeValidationFailure& e)
    {
        EXPECT_HAS_SUBSTRING(
            e.what(), std::string("While validating ONNX node '<Node(InstanceNormalization)"));
    }
    // On MacOS after we re-throw OnnxNodeValidationFailure exception, we couldn't catch it as is,
    // thus below workaround.
    catch (const std::exception& e)
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
            SERIALIZED_ZOO, "onnx/dynamic_shapes/eye_link_dyn_shape.prototxt"));
        // Should have thrown, so fail if it didn't
        FAIL() << "ONNX Importer did not detected incorrect model!";
    }
    catch (const std::exception& e)
    {
        EXPECT_HAS_SUBSTRING(e.what(),
                             std::string("While validating ONNX node '<Node(EyeLike): y"));
    }
    catch (...)
    {
        FAIL() << "The ONNX model importer failed for unexpected reason";
    }
}
