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

#include <onnx/onnx_pb.h>
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

#include "gtest/gtest.h"
#include "util/test_control.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(onnx, get_function_op_with_version)
{
    const auto* schema =
        ONNX_NAMESPACE::OpSchemaRegistry::Schema("MeanVarianceNormalization", 9, "");
    EXPECT_TRUE(schema);
    EXPECT_TRUE(schema->HasFunction());
    auto func = schema->GetFunction();
    EXPECT_EQ(func->name(), "MeanVarianceNormalization");
}

NGRAPH_TEST(onnx, check_ir_version_support)
{
    // It appears you've changed the ONNX library version used by nGraph. Please update the value
    // tested below to make sure it equals the current IR_VERSION enum value defined in ONNX headers
    //
    // You should also check the onnx_reader/onnx_model_validator.cpp file and make sure that
    // the details::onnx::is_correct_onnx_field() handles any new fields added in the new release
    // of the ONNX library. Make sure to update the "Field" enum and the function mentioned above.
    //
    // The last step is to also update the details::onnx::contains_onnx_model_keys() function
    // in the same file to make sure that prototxt format validation also covers the changes in ONNX
    EXPECT_EQ(ONNX_NAMESPACE::Version::IR_VERSION, 7)
        << "The IR_VERSION defined in ONNX does not match the version that OpenVINO supports. "
           "Please check the source code of this test for details and explanation how to proceed.";
}
