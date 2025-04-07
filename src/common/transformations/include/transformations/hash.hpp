// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/model.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @brief Hash transformation calculates hash value for ov::Model
 */
class TRANSFORMATIONS_API Hash : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("HashPass");

    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;

    /**
     * @brief Hash pass constructor
     *
     * @param output_hash_value Reference to output value. By applying hash pass on function, resulting hash value
     * will be set to this variable
     */
    Hash(uint64_t& output_hash_value);

private:
    uint64_t& m_hash;
};

}  // namespace pass
}  // namespace ov
