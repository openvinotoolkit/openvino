// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertCompressedToMixedPrecision;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertCompressedToMixedPrecision converts fp16 compressed ov::Model to mixed precision ov::Model.
 * In mixed precision ov::Models precision sensitive nodes are kept in fp32 while most of the model is in fp16.
 */
class ov::pass::ConvertCompressedToMixedPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ConvertCompressedToMixedPrecision", "0");
    bool run_on_model(const std::shared_ptr<Model>& f) override;
};
