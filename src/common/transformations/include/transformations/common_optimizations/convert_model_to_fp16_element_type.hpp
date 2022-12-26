// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertModelToFP16ElementType;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertCompressedOnlyToLegacy transformation converts compression only FP16 format to legacy FP16 format.
 */
class ov::pass::ConvertModelToFP16ElementType : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ConvertModelToFP16ElementType", "0");

    explicit ConvertModelToFP16ElementType(bool keep_precision_sensitive_in_fp32 = true)
        : m_keep_precision_sensitive_in_fp32(keep_precision_sensitive_in_fp32) {}

    bool run_on_model(const std::shared_ptr<Model>& f) override;

private:
    bool m_keep_precision_sensitive_in_fp32;
};
