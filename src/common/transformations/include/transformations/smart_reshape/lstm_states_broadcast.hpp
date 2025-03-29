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

class TRANSFORMATIONS_API LSTMStatesBroadcast;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief In case LSTMCell has constant initial hidden and cell state with single batch size
 * we make them broadcast-able by batch
 */

class ov::pass::LSTMStatesBroadcast : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("LSTMStatesBroadcast");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
