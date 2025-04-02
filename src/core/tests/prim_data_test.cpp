#include "openvino/op/identity.hpp"
#include "openvino/core/model.hpp"
#include "gtest/gtest.h"

using namespace ov;

TEST(OpIdentity, PrimDataSupport) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto identity = std::make_shared<op::v16::Identity>(input);
    auto model = std::make_shared<Model>(NodeVector{identity}, ParameterVector{input});
    ASSERT_TRUE(model != nullptr);
    ASSERT_EQ(identity->get_output_element_type(0), element::f32);
    ASSERT_EQ(identity->get_output_shape(0), Shape{2, 3});
}
