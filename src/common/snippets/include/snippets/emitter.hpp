// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <cstdint>

namespace ngraph {
namespace snippets {

using code = const uint8_t *;
using RegInfo = std::pair<std::vector<size_t>, std::vector<size_t>>;

/**
 * @interface Emitter
 * @brief Base class for all target specific code emitters used by generator.
 * @ingroup snippets
 */
class Emitter {
public:
    /**
     * @brief Default constructor
     */
    Emitter(const std::shared_ptr<ngraph::Node>& n) {
    }

    Emitter(std::vector<std::pair<std::shared_ptr<Emitter>, RegInfo>>& region) {
    }

    /**
     * @brief called by generator to generate code to produce target code for a specific operation
     * @param in vector of vector argument registers
     * @param out vector of vector resulting registers
     * @param pool optional vector of free vector registers which might be used inside method
     * @param gpr vector of free generam puproce registers which might be used inside method
     * @return void
     */
    virtual void emit_code(const std::vector<size_t>& in,
                           const std::vector<size_t>& out,
                           const std::vector<size_t>& pool = {},
                           const std::vector<size_t>& gpr  = {}) const = 0;

    /**
     * @brief called by generator to generate data section, if needed for a specific operation
     * @return void
     */
    virtual void emit_data() const {
    }
    virtual ~Emitter() = default;
};

using EmitterCode = std::pair<std::shared_ptr<Emitter>, ngraph::snippets::RegInfo>;

} // namespace snippets
} // namespace ngraph