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
     * \return Pointer to decomposed eye model.
     */
    std::shared_ptr<Node> make_eye_model(const Output<Node>& height,
                                         const Output<Node>& width,
                                         const Output<Node>& k,
                                         element::Type dtype);

    /**
     * \brief Make eye model as basic 2D eye replicated as spcified in batch size.
     *
     * \param eye    Eye model
     * \param batch  1-D tensor which defines leading batch dimensions of output eye shape.
     *
     * \return Pointer to decomposed eye model.
     */
    std::shared_ptr<Node> make_eye_batches(const Output<Node>& eye, const Output<Node>& batch);
};

}  // namespace pass
}  // namespace ov
