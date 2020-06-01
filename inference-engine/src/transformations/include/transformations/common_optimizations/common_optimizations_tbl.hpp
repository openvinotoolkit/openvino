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

// This pass must be called first in pipeline
NGRAPH_PASS(InitNodeInfo, ::ngraph::pass)
NGRAPH_PASS(ConvertPriorBox, ::ngraph::pass)  // WA: ConvertPriorBox must be executed before CF
NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
NGRAPH_PASS(RemoveFilteringBoxesBySize, ::ngraph::pass) // Resolves dynamism (replaces NonZero), CF needed
NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
NGRAPH_PASS(StridedSliceOptimization, ::ngraph::pass) // depends on CF
NGRAPH_PASS(NopElimination, ::ngraph::pass) // may introduce fake dynamism
NGRAPH_PASS(AlgebraicSimplification, ::ngraph::pass) // may introduce fake dynamism
NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
NGRAPH_PASS(ConvertScatterElementsToScatter, ::ngraph::pass) // partially depends on CF
NGRAPH_PASS(DepthToSpaceFusion, ::ngraph::pass)
