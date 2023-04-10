// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API KeepDecompressionsInFP32;

}  // namespace pass
}  // namespace ov

class ov::pass::KeepDecompressionsInFP32 : public MatcherPass {
public:
    OPENVINO_RTTI("KeepDecompressionsInFP32", "0");
    KeepDecompressionsInFP32();
};
