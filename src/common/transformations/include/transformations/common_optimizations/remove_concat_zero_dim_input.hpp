// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class OPENVINO_API RemoveConcatZeroDimInput;
class OPENVINO_API DisableRemoveConcatZeroDimInput;

/**
 * @ingroup ov_transformation_common_api
 * @brief RemoveConcatZeroDimInput transformation
 * removes input of Concat if the tensor size is equal to 0
 */

class RemoveConcatZeroDimInput : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RemoveConcatZeroDimInput");
    RemoveConcatZeroDimInput();
};

OPENVINO_API void disable_remove_concat_zerodim_input(const std::shared_ptr<Node>& node);

OPENVINO_API void enable_remove_concat_zerodim_input(const std::shared_ptr<Node>& node);

OPENVINO_API bool remove_concat_zerodim_input_is_disabled(const std::shared_ptr<Node>& node);

class DisableRemoveConcatZeroDimInput : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("DisableRemoveConcatZeroDimInput", "0", ov::RuntimeAttribute);
    DisableRemoveConcatZeroDimInput() = default;
    bool is_copyable() const override {
        return false;
    }
};

}  // namespace pass
}  // namespace ov
