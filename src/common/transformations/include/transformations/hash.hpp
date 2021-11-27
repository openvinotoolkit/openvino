// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ngraph/opsets/opset.hpp"
#include "openvino/core/function.hpp"
#include "openvino/pass/serialize.hpp"

namespace ov {
namespace pass {

/**
 * @brief Hash transformation calculates hash value for ov::Function
 */
class NGRAPH_API Hash : public ov::pass::FunctionPass {
public:
    OPENVINO_RTTI("HashPass");

    bool run_on_function(std::shared_ptr<ov::Function> f) override;

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
