// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/util.hpp"

namespace ngraph {
namespace pass {
/// \brief The DynElimination pass finds dynamic operations in a graph whose
/// shape relevant inputs have already been resolved to static values, and
/// replaces those dynamic operations with the equivalent operations using
/// static inputs and attributes.
/// \details This pass should be executed after the ConstantFolding pass.
///
/// The ConstantFolding and DynElimination passes are used together to transform
/// dynamic operations in a computation graph to static operations when the
/// graph is executed with input data.
///
/// In the example shown below, the original graph is constructed with dynamic
/// broadcast operation. When the graph is executed with input data, the input
/// shapes become available, by applying the ConstantFolding and DynElimination
/// pass, the graph is updated with dynamic broadcast being replaced by a static
/// broadcast operation.
/// <table>
/// <tr>
///     <th>Original</th>
///     <th>After %ConstantFolding</th>
///     <th>After %DynElimination</th>
/// </tr>
/// <tr>
///      <td> \image html dyn_broadcast_pre_constfld.svg </td>
///      <td> \image html dyn_broadcast_post_constfld.svg </td>
///      <td> \image html dyn_broadcast_post_dyneliminate.svg </td>
/// </tr>
/// </table>
class DynElimination : public GraphRewrite {
public:
    DynElimination();

private:
    void construct_range();
};
}  // namespace pass
}  // namespace ngraph
