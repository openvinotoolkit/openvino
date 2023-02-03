// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {
/**
 * Group of transformations which insert Identity layer in the following cases:
 * in case of eltwise sum in 16-bit input precision, one of inputs is 4 bytes, the other is 2 bytes
 * in case of eltwise mul in 16-bit input precision, both inputs are 2 bytes
 * in case of eltwise sum in low (8-bit) input precision, both inputs are 1 byte
 * in case of eltwise mul in low (8-bit) input precision, both inputs are 1 byte
 * for e sum if we have 4-4 inputs we will handle that by inserting identity activation -- handling here
 * for e sum if we have 4-2 - OK
 * for e sum if we have 2-2 inputs we need to insert diagonal
 * for e sum if we have 1-1 inputs in low precision mode - OK
 * for e mul if we have 2-2 - OK
 * for e mul if we have 1-1 in low precision mode - OK
 * for e mul if we have 2-4 - inputs we need to insert identity to put 4 bytes input into weights -- handling here
 * for e mul if we have 4-4 - inputs we need to insert 2 identities to put both 4 bytes input into weights -- handling
 * here
 */

/**
 * @brief Transformation is looking for nodes before which Identity should be inserted and mark them with appropriate rt
 * attribute
 */
class MarkIdentityCandidates : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MarkIdentityCandidates", "0");
    MarkIdentityCandidates(bool is_low_precision_input) : is_low_precision_input(is_low_precision_input) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;

private:
    bool is_low_precision_input;
};

/**
 * @brief Transformation inserts Identity layer based on rt attribute
 */
class InsertIdentity : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("InsertIdentity", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};

/**
 * @brief In cases that network output layer is connected to only one layer which is activation additional identity is
 * inserted so the operation is not fused with the activation allowing to get the results from said layer
 */
class BreakFusingOfOutputLayers : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BreakFusingOfOutputLayers", "0");
    BreakFusingOfOutputLayers();
};

/**
 * @brief IdentityCandidates removes attribute mark for identity insertion
 */
class IdentityCandidatesCleanup : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("IdentityCandidatesCleanup", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
