// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/negative.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, unary_arithmetic_bad_argument_element_types) {
    auto tv0_2_4_param = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 4});
    try {
        auto bc = make_shared<op::v0::Negative>(tv0_2_4_param);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Arguments cannot have boolean element type"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
