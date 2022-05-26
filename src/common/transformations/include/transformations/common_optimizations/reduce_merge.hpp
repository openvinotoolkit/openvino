// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ReduceMerge;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ReduceMerge transformation matches following graph:
 *
 *    +----------+         +----------+
 *    |    A     |         |    B     |
 *    +----------+         +----------+
 *         |                    |
 *         ---------    ---------
 *                 |    |
 *                 v    v
 *               +--------+   +--------+
 *               | Reduce |   |    C   |
 *               +--------+   +--------+
 *                   |             |
 *                   |       -------
 *                   |       |
 *                   v       v
 *                  +----------+
 *                  |  Reduce  |
 *                  +----------+
 *
 *
 * and replaces with:
 *
 *           +----------+     +----------+
 *           |    B     |     |    C     |
 *           +----------+     +----------+
 *                |                |
 *                -------    -------
 *                      |    |
 *                      v    v
 *    +----------+   +----------+
 *    |     A    |   |  Concat  |
 *    +----------+   +----------+
 *          |             |
 *          |      --------
 *          |      |
 *          v      v
 *        +----------+
 *        |  Reduce  |
 *        +----------+
 *
 */
class ngraph::pass::ReduceMerge : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReduceMerge", "0");
    ReduceMerge();
};
