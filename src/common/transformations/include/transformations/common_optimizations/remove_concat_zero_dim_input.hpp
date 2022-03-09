// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RemoveConcatZeroDimInput;
class TRANSFORMATIONS_API DisableRemoveConcatZeroDimInput;

/**
 * @ingroup ie_transformation_common_api
 * @brief RemoveConcatZeroDimInput transformation
 * removes input of Concat if the tensor size is equal to 0
 */

class RemoveConcatZeroDimInput : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    RemoveConcatZeroDimInput();
};

TRANSFORMATIONS_API void disable_remove_concat_zerodim_input(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void enable_remove_concat_zerodim_input(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool remove_concat_zerodim_input_is_disabled(const std::shared_ptr<Node>& node);

class DisableRemoveConcatZeroDimInput : public ov::RuntimeAttribute {
public:
    NGRAPH_RTTI_DECLARATION;
    DisableRemoveConcatZeroDimInput() = default;
    bool is_copyable() const override {
        return false;
    }
};

}  // namespace pass
}  // namespace ov