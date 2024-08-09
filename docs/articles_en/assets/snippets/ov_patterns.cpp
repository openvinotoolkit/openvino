// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ! [ov:imports]
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::pass;
using namespace std;
// ! [ov:imports]

// ! [ov:create_simple_model_and_pattern]
void create_simple_model_and_pattern() {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a sample model
    auto pattern_mul = std::make_shared<ov::op::v0::MatMul>(pattern::any_input(), pattern::any_input(), false, false);
    auto pattern_abs = std::make_shared<ov::op::v0::Abs>(pattern_mul->output(0));
    auto pattern_relu = std::make_shared<ov::op::v0::Relu>(pattern_abs->output(0));

    // pattern_relu should perfectly match model_relu
}
// ! [ov:create_simple_model_and_pattern]


// ! [ov:create_simple_model_and_pattern_wrap_type]
void create_simple_model_and_pattern_wrap_type() {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a sample model
    auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
    auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});

    // pattern_relu should perfectly match model_relu
}
// ! [ov:create_simple_model_and_pattern_wrap_type]


// ! [ov:wrap_type_list]
void wrap_type_list() {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));
    auto model_sig = std::make_shared<ov::op::v0::Sigmoid>(model_abs->output(0));
    auto model_result1 = std::make_shared<ov::op::v0::Result>(model_sig->output(0));

    // Create a sample model
    auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
    auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu, ov::op::v0::Sigmoid>({pattern_abs->output(0)});

    // pattern_relu should perfectly matches model_relu and model_sig
}
// ! [ov:wrap_type_list]

void patterns_misc() {
{
    // ! [ov:any_input]
        auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
        auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
        auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});
    // ! [ov:any_input]

    // ! [ov:wrap_type_predicate]
        ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern::any_input()}, pattern::consumers_count(2));
    // ! [ov:wrap_type_predicate]
}
{
    // ! [ov:any_input_predicate]
        auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input([](const Output<Node>& value){
                                                                                return value.get_shape().size() == 4;}),
                                                                            pattern::any_input([](const Output<Node>& value){
                                                                                return value.get_shape().size() == 4;})});
        auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
        auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});
    // ! [ov:any_input_predicate]


    // ! [ov:optional_predicate]
        auto pattern_sig_opt = ov::pass::pattern::optional<ov::op::v0::Sigmoid>(pattern_relu, pattern::consumers_count(2));
    // ! [ov:optional_predicate]
}
}


// ! [ov:pattern_or]
void pattern_or() {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a red branch
    auto red_pattern_add = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto red_pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({red_pattern_add->output(0)});
    auto red_pattern_sigmoid = ov::pass::pattern::wrap_type<ov::op::v0::Sigmoid>({red_pattern_relu->output(0)});

    // Create a blue branch
    auto blue_pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto blue_pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({blue_pattern_mul->output(0)});
    auto blue_pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({blue_pattern_abs->output(0)});

    // Create Or node
    auto pattern_or = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{red_pattern_sigmoid->output(0), blue_pattern_relu->output(0)});

    // pattern_or should perfectly matches model_relu
}
// ! [ov:pattern_or]


// ! [ov:pattern_optional_middle]
void pattern_optional_middle() {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a sample pattern with an Optional node in the middle
    auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
    auto pattern_sig_opt = ov::pass::pattern::optional<ov::op::v0::Sigmoid>({pattern_abs->output(0)});
    auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_sig_opt->output(0)});

    // pattern_relu should perfectly match model_relu
}
// ! [ov:pattern_optional_middle]


// ! [ov:pattern_optional_top]
void pattern_optional_top() {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a sample pattern an optional top node
    auto pattern_sig_opt = ov::pass::pattern::optional<ov::op::v0::Sigmoid>(pattern::any_input());
    auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern_sig_opt, pattern::any_input()});
    auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
    auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});

    // pattern_relu should perfectly match model_relu
}
// ! [ov:pattern_optional_top]


// ! [ov:pattern_optional_root]
void pattern_optional_root() {
    // Create a sample model
    PartialShape shape{2, 2};
    auto model_param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_param2 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_add = std::make_shared<ov::op::v1::Add>(model_param1->output(0), model_param2->output(0));
    auto model_param3 = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto model_mul = std::make_shared<ov::op::v0::MatMul>(model_add->output(0), model_param3->output(0), false, false);
    auto model_abs = std::make_shared<ov::op::v0::Abs>(model_mul->output(0));
    auto model_relu = std::make_shared<ov::op::v0::Relu>(model_abs->output(0));
    auto model_result = std::make_shared<ov::op::v0::Result>(model_relu->output(0));

    // Create a sample pattern an optional top node
    auto pattern_mul = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({pattern::any_input(), pattern::any_input()});
    auto pattern_abs = ov::pass::pattern::wrap_type<ov::op::v0::Abs>({pattern_mul->output(0)});
    auto pattern_relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({pattern_abs->output(0)});
    auto pattern_sig_opt = ov::pass::pattern::optional<ov::op::v0::Sigmoid>(pattern_relu);

    // pattern_relu should perfectly match model_relu
}
// ! [ov:pattern_optional_root]