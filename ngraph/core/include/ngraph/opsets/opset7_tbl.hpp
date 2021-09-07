// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NGRAPH_OP
#    warning "NGRAPH_OP not defined"
#    define NGRAPH_OP(x, y)
#endif

#define OPENVINO_OP NGRAPH_OP
#include "openvino/opsets/opset7_tbl.hpp"
#undef OPENVINO_OP
