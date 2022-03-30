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
    RUN_ON_FUNCTION_SCOPE(AssignRegisters);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::AssignRegisters")
//    int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
//    std::vector<int64_t> available_registers {0, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<int64_t> available_registers {0, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15};
    using Reg = size_t;
    auto ops = f->get_ordered_ops();
    decltype(ops) stmts;
    std::copy_if(ops.begin(), ops.end(), std::back_inserter(stmts), [](decltype(ops[0]) op) {
        return !(std::dynamic_pointer_cast<opset1::Parameter>(op) || std::dynamic_pointer_cast<opset1::Result>(op));
        });

    size_t rdx = 0;
    std::map<std::shared_ptr<descriptor::Tensor>, Reg> regs;
    for (auto op : stmts) {
        for (auto output : op->outputs()) {
            regs[output.get_tensor_ptr()] = rdx++;
        }
    }

    std::vector<std::set<Reg>> used;
    std::vector<std::set<Reg>> def;

    for (auto op : stmts) {
        std::set<Reg> u;
        for (auto input : op->inputs()) {
            if (regs.count(input.get_tensor_ptr())) {
                u.insert(regs[input.get_tensor_ptr()]);
            }
        }
        used.push_back(u);

        std::set<Reg> d;
        if (!std::dynamic_pointer_cast<snippets::op::Store>(op)) {
            for (auto output : op->outputs()) {
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
                for (auto out : node->outputs()) {
                    for (auto port : out.get_target_inputs()) {
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
        int ln = lifeIn.size()-1;
        for (auto& x : lifeIn) {
            if (x.find(i) != x.end()) {
                return ln;
            }
            ln--;
        }
        return i;
    };

    for (size_t i = 0; i < stmts.size(); i++) {
        live_intervals.insert(std::make_pair(i, find_last_use(i)));
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

    for (auto reg : regs) {
        physical_regs[reg.first] = register_map[reg.second];
    }

    size_t constantID = 0;

    for (auto n : f->get_ordered_ops()) {
        auto& rt = n->get_rt_info();
        // nothing to do for model signature
        if (std::dynamic_pointer_cast<opset1::Parameter>(n) || std::dynamic_pointer_cast<opset1::Result>(n)) {
            continue;
        }

        // store only effective address
        if (auto result = std::dynamic_pointer_cast<snippets::op::Store>(n)) {
//            auto ea = reg64_tmp_start+static_cast<int64_t>(f->get_result_index(result) + f->get_parameters().size());
            auto ea = available_registers[static_cast<int64_t>(f->get_result_index(result) + f->get_parameters().size())];
            rt["effectiveAddress"] = ea;
            continue;
        }
        // store effective address and procced with vector registers
        if (ov::as_type_ptr<ngraph::snippets::op::Load>(n) || ov::as_type_ptr<ngraph::snippets::op::BroadcastLoad>(n)) {
            auto source = n->get_input_source_output(0).get_node_shared_ptr();

            if (auto param = ov::as_type_ptr<opset1::Parameter>(source)) {
//                auto ea = reg64_tmp_start+static_cast<int64_t>(f->get_parameter_index(param));
                auto ea = available_registers[static_cast<int64_t>(f->get_parameter_index(param))];
                rt["effectiveAddress"] = ea;
            } else if (auto constant = ov::as_type_ptr<opset1::Constant>(source)) {
//                auto ea = reg64_tmp_start+static_cast<int64_t>(f->get_parameters().size() + f->get_results().size() + 1 + constantID);
                auto ea = available_registers[static_cast<int64_t>(f->get_parameters().size() + f->get_results().size() + 1 + constantID)];
                rt["effectiveAddress"] = ea;
                constantID++;
            } else {
                throw ngraph_error("load/broadcast should follow only Parameter or non-Scalar constant");
            }
        }

        std::vector<size_t> regs; regs.reserve(n->outputs().size());
        for (auto output : n->outputs()) {
            auto allocated = physical_regs[output.get_tensor_ptr()];
            regs.push_back(allocated);
        }
        rt["reginfo"] = regs;
    }

    return false;
}
