// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/negative.hpp"

using namespace std;
using namespace ov;
using testing::HasSubstr;

TEST(type_prop, unary_arithmetic_unsupported_input_element_types) {
    {
        const auto param = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 4});
        OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Negative>(param),
                        ov::NodeValidationFailure,
                        HasSubstr("This operation does not support input with element type: boolean"));
    }
    {
        const auto param = make_shared<ov::op::v0::Parameter>(element::string, Shape{2, 4});
        OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Negative>(param),
                        ov::NodeValidationFailure,
                        HasSubstr("This operation does not support input with element type: string"));
    }
}
