// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Current version of ModelOptimizer substitutes SoftSign activation
 * function with next subgraph
 *  a layer
 *  |  \
 * abs  \
 *  |    |
 * add   |
 *  |    |
 * power |
 *  \   /
 *  Divide
 *
 * See model-optimizer/extensions/front/softsign_replacer.py
 *
 * The ConvertDivide transformation from CommonOptimizations
 * substitutes Divide with {-1} and add constant {1}
 * - GNA supports Power [0, 2.8]
 * - Add, Power, Divide layers are more perfomance expensive in GNA
 *   than SoftSign PWL
 *
 * Legacy SubstituteSoftSignPass supports irv7 model where SoftSign subgraph
 * could have been without add layer. Current ModelOptimezer always generates
 * SoftSign subgraph with that layer.
 *
 * SubstituteSoftsign transformation does backward substitution to SoftSign.
 * TODO: remove that pass as soon as ModelOptimizer will not substitute SoftSign activation
 */
class SubstituteSoftsign : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SubstituteSoftsign", "0");
    SubstituteSoftsign();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
