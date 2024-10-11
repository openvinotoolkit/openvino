// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertConvertPromoteTypes;

}  // namespace pass
}  // namespace ov

/// \brief Transformation to replace ConvertPromoteTypes with pair of Convert ops for each input to evaluated common
/// element type.
class ov::pass::ConvertConvertPromoteTypes : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertConvertPromoteTypes", "0");
    ConvertConvertPromoteTypes();
};
