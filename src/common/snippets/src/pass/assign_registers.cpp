// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include <openvino/cc/selective_build.h>
#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/assign_registers.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>

#include <iterator>

bool ngraph::snippets::pass::AssignRegisters::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(AssignRegisters);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::AssignRegisters")
    using Reg = size_t;
    auto ops = f->get_ordered_ops();
    decltype(ops) stmts;
    std::copy_if(ops.begin(), ops.end(), std::back_inserter(stmts), [](decltype(ops[0]) op) {
        return !(std::dynamic_pointer_cast<opset1::Parameter>(op) || std::dynamic_pointer_cast<opset1::Result>(op));
        });

    size_t rdx = 0;
    std::map<std::shared_ptr<descriptor::Tensor>, Reg> regs;
    for (const auto& op : stmts) {
        for (const auto& output : op->outputs()) {
            regs[output.get_tensor_ptr()] = rdx++;
        }
    }

    std::vector<std::set<Reg>> used;
    std::vector<std::set<Reg>> def;

    for (const auto& op : stmts) {
        std::set<Reg> u;
        for (const auto& input : op->inputs()) {
            if (regs.count(input.get_tensor_ptr())) {
                u.insert(regs[input.get_tensor_ptr()]);
            }
        }
        used.push_back(u);

        std::set<Reg> d;
        if (!std::dynamic_pointer_cast<snippets::op::Store>(op)) {
            for (const auto& output : op->outputs()) {
                d.insert(regs[output.get_tensor_ptr()]);
            }
        }
        def.push_back(d);
    }

    // define life intervals
    std::vector<std::set<Reg>> lifeIn(stmts.size(), std::set<Reg>());
    std::vector<std::set<Reg>> lifeOut(stmts.size(), std::set<Reg>());

    for (size_t i = 0; i < stmts.size(); i++) {
        for (size_t n = 0; n < stmts.size(); n++) {
            std::set_difference(lifeOut[n].begin(), lifeOut[n].end(), def[n].begin(), def[n].end(), std::inserter(lifeIn[n], lifeIn[n].begin()));
            lifeIn[n].insert(used[n].begin(), used[n].end());
        }
        for (size_t n = 0; n < stmts.size(); n++) {
            auto node = stmts[n];
            if (!std::dynamic_pointer_cast<snippets::op::Store>(node)) {
                for (const auto& out : node->outputs()) {
                    for (const auto& port : out.get_target_inputs()) {
                        auto pos = std::find(stmts.begin(), stmts.end(), port.get_node()->shared_from_this());
                        if (pos != stmts.end()) {
                            auto k = pos-stmts.begin();
                            lifeOut[n].insert(lifeIn[k].begin(), lifeIn[k].end());
                        }
                    }
                }
            }
        }
    }

    struct by_starting {
        auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
            return lhs.first < rhs.first|| (lhs.first == rhs.first && lhs.second < rhs.second);
        }
    };

    struct by_ending {
        auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
            return lhs.second < rhs.second || (lhs.second == rhs.second && lhs.first < rhs.first);
        }
    };

    std::set<std::pair<int, int>, by_starting> live_intervals;

    std::reverse(lifeIn.begin(), lifeIn.end());
    auto find_last_use = [lifeIn](int i) -> int {
        int ln = static_cast<int>(lifeIn.size()) - 1;
        for (auto& x : lifeIn) {
            if (x.find(i) != x.end()) {
                return ln;
            }
            ln--;
        }
        return i;
    };

    for (size_t i = 0; i < stmts.size(); i++) {
        live_intervals.insert(std::make_pair(static_cast<int>(i), find_last_use(static_cast<int>(i))));
    }

    // http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf
    std::multiset<std::pair<int, int>, by_ending> active;
    std::map<Reg, Reg> register_map;
    std::stack<Reg> bank;
    for (int i = 0; i < 16; i++) bank.push(16-1-i);

    for (auto interval : live_intervals) {
        // check expired
        while (!active.empty()) {
            auto x = *active.begin();
            if (x.second >= interval.first) {
                break;
            }
            active.erase(x);
            bank.push(register_map[x.first]);
        }
        // allocate
        if (active.size() == 16) {
            throw ngraph_error("caanot allocate registers for a snippet ");
        } else {
            register_map[interval.first] = bank.top();
            bank.pop();
            active.insert(interval);
        }
    }

    std::map<std::shared_ptr<descriptor::Tensor>, Reg> physical_regs;

    for (const auto& reg : regs) {
        physical_regs[reg.first] = register_map[reg.second];
    }
    const auto num_parameters = f->get_parameters().size();
    for (const auto& n : f->get_ordered_ops()) {
        auto& rt = n->get_rt_info();
        std::vector<size_t> regs;
        regs.reserve(n->outputs().size());
        /* The main idea here is that each operation stores its output regs in rt["reginfo"]. Input and output regs are
         * then derived by parsing node's and parent's rt["reginfo"], look into ngraph::snippets::getRegisters for details.
         * Note also that Parameter and Result store general-purpose register index, because they work with memory
         * (memory pointer is stored in gpr). All other "regular" ops store vector regs indexes, since calculations are
         * performed on registers.
         */
        if (is_type<ov::op::v0::Result>(n)) {
            continue;
        } else if (const auto& param = ov::as_type_ptr<ov::op::v0::Parameter>(n)) {
            regs.push_back(f->get_parameter_index(param));
        } else if (const auto& store = ov::as_type_ptr<ngraph::snippets::op::Store>(n)) {
            regs.push_back(f->get_result_index(store) + num_parameters);
        } else {
            for (const auto& output : n->outputs()) {
                auto allocated = physical_regs[output.get_tensor_ptr()];
                regs.push_back(allocated);
            }
        }
        rt["reginfo"] = regs;
    }

    return false;
}
