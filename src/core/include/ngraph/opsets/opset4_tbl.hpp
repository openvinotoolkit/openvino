// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NGRAPH_OP
#ifndef _REGISTER_NGRAPH_OP
#    warning "NGRAPH_OP not defined"
#    define NGRAPH_OP(x, y)
#else
#    define _NGRAPH_AUTO_OPSET_REGISTRATOR
#    define NGRAPH_OP(x, y) _REGISTER_NGRAPH_OP(opset4, x, y)
#endif
#endif

#define _OPENVINO_OP_REG NGRAPH_OP
#include "openvino/opsets/opset4_tbl.hpp"
#undef _OPENVINO_OP_REG

#ifdef _NGRAPH_AUTO_OPSET_REGISTRATOR
#undef _NGRAPH_AUTO_OPSET_REGISTRATOR
#undef NGRAPH_OP
#endif
