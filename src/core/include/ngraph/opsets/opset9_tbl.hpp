// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NGRAPH_OP
#    warning "NGRAPH_OP not defined"
#    define NGRAPH_OP(x, y)
#endif

#define _OPENVINO_OP_REG NGRAPH_OP
#include "openvino/opsets/opset9_tbl.hpp"
#undef _OPENVINO_OP_REG
