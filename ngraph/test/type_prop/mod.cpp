#include "arithmetic_ops.hpp"

using Type = ::testing::Types<ngraph::op::v1::Mod>;

INSTANTIATE_TYPED_TEST_CASE_P(type_prop_mod, ArithmeticOperator, Type);