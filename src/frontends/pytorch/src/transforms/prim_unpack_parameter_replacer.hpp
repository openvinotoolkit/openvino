// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

// This transformation replaces all prim::TupleUnpack/prim::ListUnpack operations coming after Parameters
// with more Parameters -- one new parameter for each unpacked output. The original Parameter
// is replaced with these new Parameters preserving the order relative to other Parameters in a model.
// Order of new parameters is the same as the order of unpacked outputs.
// If unpack operation has a consumer that is also unpack operation, the transformation applies
// the replacement recursively until all unpack operations that take a Parameter output are eliminated.
//
// For example, if a model has the following signature: a, (b, (c, d)), e, where a, b, c, d, and e are
// tensors, and (x1, x2) means tuple consisting two elements x1 and x2, then the resulting model
// after the transformation will have a, b, c, d, e as inputs (without tuples, flattened).
// Note, that there is no special 'tuple' type of an input, tuple structure is restored by
// following prim::TupleUnpack operations in the graph only assuming that they can be applied on
// tuples only and the most nested objects in those tuples are tensors.
class DecomposeUnpackParameters : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::pytorch::pass::DecomposeUnpackParameters");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
