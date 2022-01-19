// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef INSERT_RESHAPE_AROUND_MATMUL_HPP
#define INSERT_RESHAPE_AROUND_MATMUL_HPP

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

// @brief Insert Reshapes from 3d/4d to 2d before MatMul and from 2d to 3d/4d after MatMul
class InsertReshapeAroundMatmul : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    InsertReshapeAroundMatmul();
};

class InsertReshapeAroundMatmulWithAdd : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    InsertReshapeAroundMatmulWithAdd();
};

class InsertReshapeAroundMatmulWithFq : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    InsertReshapeAroundMatmulWithFq();
};

class InsertReshapeAroundMatmulWithTranspose : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    InsertReshapeAroundMatmulWithTranspose();
};

} // namespace GNAPluginNS

#endif // INSERT_RESHAPE_AROUND_MATMUL_HPP
