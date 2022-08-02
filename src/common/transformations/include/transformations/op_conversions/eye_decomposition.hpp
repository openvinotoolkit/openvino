// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * \ingroup ie_transformation_common_api
 *
 * \brief Do eye decomposition to sub-graph (model).
 */
class TRANSFORMATIONS_API EyeDecomposition : public MatcherPass {
public:
    OPENVINO_RTTI("EyeDecomposition", "0");
    EyeDecomposition();

protected:
    /**
     * \brief Make eye model which generate eye matrix.
     *
     * If 'k' is outside the eye dimension then result matrix will be filled with zeros.
     *
     * \param height  Height of eye
     * \param width   Width of eye
     * \param k       Eye diagonal shift.
     * \param dtype   Data type of eye.
     *
     * \return std::shared_ptr<Node> to decomposed eye model.
     */
    std::shared_ptr<Node> make_eye_model(const Output<Node>& height,
                                         const Output<Node>& width,
                                         const Output<Node>& k,
                                         element::Type dtype);
};

}  // namespace pass
}  // namespace ov
