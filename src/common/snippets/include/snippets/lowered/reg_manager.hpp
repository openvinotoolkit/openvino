// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "snippets/emitter.hpp"
#include "snippets/generator.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/kernel.hpp"

/**
 * @interface RegManager
 * @brief The class holds supplementary info about assigned registers and live ranges
 * @ingroup snippets
 */

namespace ov::snippets::lowered {

// LiveInterval is a pair of (start, stop) expression execution numbers, where:
// start - exec number of the expression that produced the value
// stop - exec number of the last consumer of the value
using LiveInterval = std::pair<decltype(Expression().get_exec_num()), decltype(Expression().get_exec_num())>;
class RegManager {
public:
    RegManager() = delete;
    explicit RegManager(const std::shared_ptr<const Generator>& generator) : m_generator(generator) {}
    [[nodiscard]] RegType get_reg_type(const ov::Output<Node>& out) const {
        return m_generator->get_op_out_reg_type(out);
    }
    [[nodiscard]] std::vector<Reg> get_vec_reg_pool() const {
        return m_generator->get_target_machine()->get_vec_reg_pool();
    }

    void set_live_range(const Reg& reg, const LiveInterval& interval) {
        OPENVINO_ASSERT(m_reg_live_range.insert({reg, interval}).second, "Live range for this reg is already set");
    }

    [[nodiscard]] std::vector<Reg> get_kernel_call_regs(const std::shared_ptr<snippets::op::Kernel>& kernel) const {
        const auto& abi_regs = m_generator->get_target_machine()->get_abi_arg_regs();
        const auto num_kernel_args = kernel->get_num_call_args();
        assert(abi_regs.size() > num_kernel_args && "Too many kernel args requested");
        return {abi_regs.begin(), abi_regs.begin() + static_cast<int64_t>(num_kernel_args)};
    }

    [[nodiscard]] std::vector<Reg> get_gp_regs_except_kernel_call(
        const std::shared_ptr<snippets::op::Kernel>& kernel) const {
        auto res = m_generator->get_target_machine()->get_gp_reg_pool();
        std::set<Reg> kernel_call;
        for (auto r : get_kernel_call_regs(kernel)) {
            kernel_call.insert(r);
        }
        res.erase(std::remove_if(res.begin(),
                                 res.end(),
                                 [&kernel_call](const Reg& r) {
                                     return kernel_call.count(r) != 0;
                                 }),
                  res.end());
        return res;
    }

    [[nodiscard]] const LiveInterval& get_live_range(const Reg& reg) const {
        assert(m_reg_live_range.count(reg) && "Live range for this reg was not set");
        return m_reg_live_range.at(reg);
    }
    [[nodiscard]] const std::map<Reg, LiveInterval>& get_live_range_map() const {
        return m_reg_live_range;
    }

private:
    // Maps Register to {Start, Stop} pairs
    std::map<Reg, LiveInterval> m_reg_live_range;
    const std::shared_ptr<const Generator> m_generator;
};

}  // namespace ov::snippets::lowered
