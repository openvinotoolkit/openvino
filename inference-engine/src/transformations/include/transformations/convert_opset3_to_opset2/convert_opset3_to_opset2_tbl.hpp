// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NGRAPH_PASS
#warning "NGRAPH_PASS is not defined"
#define NGRAPH_PASS(A, B)
#endif

// To register new pass you need to define NGRAPH_PASS
// Usage example:
//   ngraph::pass:Manager pm;
//   #define NGRAPH_PASS(NAME, NAMESPACE)   pm.register_pass<NAMESPACE::NAME>();
//   #include <transformations/transformations_tbl.hpp>
//   #undef NGRAPH_PASS

NGRAPH_PASS(ConvertBroadcast3, ::ngraph::pass)
NGRAPH_PASS(ConvertNMS1ToNMS3, ::ngraph::pass)
NGRAPH_PASS(ConvertShapeOf3, ::ngraph::pass)
NGRAPH_PASS(ConvertShuffleChannels3, ::ngraph::pass)
NGRAPH_PASS(ConvertTopK3, ::ngraph::pass)
