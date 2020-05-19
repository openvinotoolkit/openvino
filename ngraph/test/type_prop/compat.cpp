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
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

class CompatOp : public ngraph::op::Op
{
public:
    static constexpr NodeTypeInfo type_info{"CompatOp", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    CompatOp() = default;

    CompatOp(const Output<Node>& value)
        : Op({value})
    {
    }

    // Test for API compatibility
    bool visit_attributes(AttributeVisitor& visitor) override { return true; }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override
    {
        return make_shared<CompatOp>(new_args.at(0));
    }
    void validate_and_infer_types() override
    {
        auto arg = input_value(0);
        set_output_type(0, arg.get_element_type(), arg.get_shape());
    }

protected:
    // Deprecated method
    virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                   const OutputVector& deltas) override
    {
        adjoints.add_delta(input_value(0), input_value(0) * deltas.at(0));
    }
};

constexpr NodeTypeInfo CompatOp::type_info;

TEST(compat, node)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{10});
    auto c = make_shared<op::Parameter>(element::f32, Shape{10});
    auto x = make_shared<CompatOp>(param);
    auto result = make_shared<op::Result>(x);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{param});
    autodiff::Adjoints adjoints({result}, {c});
    auto bprop = adjoints.backprop_output(param);
    ASSERT_TRUE(bprop.get_index() == 0);
    ASSERT_TRUE(is_type<op::v0::Multiply>(bprop.get_node_shared_ptr()));
    set<Output<Node>> params;
    params.insert(bprop.get_node_shared_ptr()->input_value(0));
    params.insert(bprop.get_node_shared_ptr()->input_value(1));
    EXPECT_TRUE(params.count(param) == 1);
    EXPECT_TRUE(params.count(c) == 1);
}
