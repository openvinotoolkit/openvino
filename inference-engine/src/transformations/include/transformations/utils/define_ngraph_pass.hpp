// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NGRAPH_PASS(NAME, NAMESPACE) \
transforms.push_back(manager.register_pass<NAMESPACE::NAME>());

#define REGISTER_GRAPH_REWRITE_PASS(A) \
class A : public ngraph::pass::GraphRewrite { public: A() : GraphRewrite() {} }; \
transforms.push_back(manager.register_pass<A>()); \
if (auto pass = std::dynamic_pointer_cast<ngraph::pass::GraphRewrite>(transforms.back())) { \
    anchor = pass; \
} else { throw ngraph::ngraph_error(""); }

#define REGISTER_MATCHER(NAME, NAMESPACE) \
auto NAME = std::make_shared<NAMESPACE::NAME>(); \
if (auto t_param = std::dynamic_pointer_cast<PassParam>(NAME)) { \
    t_param->setCallback(transformation_callback); \
} \
anchor->copy_matchers(NAME);
