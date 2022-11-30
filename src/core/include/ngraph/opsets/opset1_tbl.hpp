// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This collection contains one entry for each op. If an op is added it must be
// added to this list.
//
// In order to use this list you want to define a macro named exactly NGRAPH_OP
// When you are done you should undef the macro
// As an example if you wanted to make a list of all op names as strings you could do this:
//
// #define NGRAPH_OP(a,b) #a,
// std::vector<std::string> op_names{
// #include "this include file name"
// };
// #undef NGRAPH_OP
//
// This sample expands to a list like this:
// "Abs",
// "Acos",
// ...
//
// #define NGRAPH_OP(a,b) b::a,
// std::vector<std::string> op_names{
// #include "this include file name"
// };
// #undef NGRAPH_OP
//
// This sample expands to a list like this:
// ngraph::op::Abs,
// ngraph::op::Acos,
// ...
//
// It's that easy. You can use this for fun and profit.

#ifndef NGRAPH_OP
#ifndef _REGISTER_NGRAPH_OP
#    warning "NGRAPH_OP not defined"
#    define NGRAPH_OP(x, y)
#else
#    define _NGRAPH_AUTO_OPSET_REGISTRATOR
#    define NGRAPH_OP(x, y) _REGISTER_NGRAPH_OP(opset1, x, y)
#endif
#endif

#define _OPENVINO_OP_REG NGRAPH_OP
#include "openvino/opsets/opset1_tbl.hpp"
#undef _OPENVINO_OP_REG

#ifdef _NGRAPH_AUTO_OPSET_REGISTRATOR
#undef _NGRAPH_AUTO_OPSET_REGISTRATOR
#undef NGRAPH_OP
#endif
