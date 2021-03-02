// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NGRAPH_OP
#warning "NGRAPH_OP not defined"
#define NGRAPH_OP(x, y)
#endif

NGRAPH_OP(PSROIPooling, ngraph::op::v0)
NGRAPH_OP(ROIPooling, ngraph::op::v0)

NGRAPH_OP(ROIAlign, ngraph::op::v3)
