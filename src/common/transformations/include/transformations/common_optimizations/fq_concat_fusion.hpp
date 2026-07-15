// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FakeQuantizeConcatFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief FakeQuantizeConcatFusion replaces identical FakeQuantize operations on all
 * inputs of a Concat followed by an identical output FakeQuantize with a single
 * FakeQuantize after Concat.
 *
 * Before:
 *
 *   data_0 -> FQ(a,b,c,d,l) -\
 *                             \
 *   data_1 -> FQ(a,b,c,d,l) ----> Concat -> FQ(a,b,c,d,l) -> next
 *                             /
 *   data_2 -> FQ(a,b,c,d,l) -/
 *
 * After:
 *
 *   data_0 -------------------\
 *                              \
 *   data_1 ---------------------> Concat -> FQ(a,b,c,d,l) -> next
 *                              /
 *   data_2 -------------------/
 */
class ov::pass::FakeQuantizeConcatFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FakeQuantizeConcatFusion");
    FakeQuantizeConcatFusion();
};
