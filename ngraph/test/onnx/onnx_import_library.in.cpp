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
#include "util/all_close.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(onnx_${BACKEND_NAME}, get_function_op_with_version)
{
    const auto* schema =
        ONNX_NAMESPACE::OpSchemaRegistry::Schema("MeanVarianceNormalization", 9, "");
    EXPECT_TRUE(schema);
    EXPECT_TRUE(schema->HasFunction());
    auto func = schema->GetFunction();
    EXPECT_EQ(func->name(), "MeanVarianceNormalization");
}
