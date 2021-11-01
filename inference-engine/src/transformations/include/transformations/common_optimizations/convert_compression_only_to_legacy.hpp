// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformations_visibility.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertCompressedOnlyToLegacy;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertCompressedOnlyToLegacy transformation converts compression only FP16 format to legacy FP16 format.
 */

class ov::pass::ConvertCompressedOnlyToLegacy : public ov::pass::FunctionPass {
public:
    OPENVINO_RTTI("ConvertCompressedOnlyToLegacy", "0");
    bool run_on_function(std::shared_ptr<Function> f) override;
};

