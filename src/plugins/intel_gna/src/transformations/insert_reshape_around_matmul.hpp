// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef INSERT_RESHAPE_AROUND_MATMUL_HPP
#define INSERT_RESHAPE_AROUND_MATMUL_HPP

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

// @brief Insert Reshapes from 3d/4d to 2d before MatMul and from 2d to 3d/4d after MatMul
class InsertReshapeAroundMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertReshapeAroundMatmul", "0");
    InsertReshapeAroundMatmul();
};

class InsertReshapeAroundMatmulWithAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertReshapeAroundMatmulWithAdd", "0");
    InsertReshapeAroundMatmulWithAdd();
};

class InsertReshapeAroundMatmulWithFq : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertReshapeAroundMatmulWithFq", "0");
    InsertReshapeAroundMatmulWithFq();
};

class InsertReshapeAroundMatmulWithTranspose : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertReshapeAroundMatmulWithTranspose", "0");
    InsertReshapeAroundMatmulWithTranspose();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov

#endif  // INSERT_RESHAPE_AROUND_MATMUL_HPP
