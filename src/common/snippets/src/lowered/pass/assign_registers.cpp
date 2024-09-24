// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/assign_registers.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"

// This header is needed to avoid MSVC warning "C2039: 'inserter': is not a member of 'std'"
#include <iterator>

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void AssignRegisters::set_reg_types(LinearIR& linear_ir) {
    for (const auto& expr : linear_ir) {
        const auto op = expr->get_node();
        if (ov::is_type<op::LoopEnd>(op) ||
            ov::is_type<ov::op::v0::Result>(op)
#ifdef SNIPPETS_DEBUG_CAPS
        || ov::is_type<op::PerfCountBeginBase>(op)
        || ov::is_type<op::PerfCountEndBase>(op)
#endif
        )
        continue;

        OPENVINO_ASSERT(expr->get_output_count() == op->get_output_size(), "Incorrect count of output port descriptors!");
        for (size_t i = 0; i < expr->get_output_count(); ++i) {
            const auto reg_type = m_reg_type_mapper(op->output(i));
            expr->get_output_port_descriptor(i)->set_reg_type(reg_type);
            // propogate to consumers
            for (const auto& consumer : expr->get_output_port_connector(i)->get_consumers()) {
                consumer.get_descriptor_ptr()->set_reg_type(reg_type);
            }
        }
    }
}

bool AssignRegisters::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AssignRegisters")
    using Reg = size_t;
    using tensor = PortConnectorPtr;

    set_reg_types(linear_ir);
    const auto& exprs = linear_ir.get_ops();
    const auto& params = linear_ir.get_parameters();
    const auto& results = linear_ir.get_results();
    Reg num_expressions = exprs.size();
    Reg num_parameters = params.size();
    Reg num_results = results.size();

    size_t io_index = 0;
    // Define a set of immune tensors that will be ignored by auto reg allocation => their reg allocation is done manually
    std::map<tensor, Reg> manually_assigned_gprs, manually_assigned_vecs;
    for (const auto& param : params) {
        manually_assigned_gprs[param->get_output_port_connector(0)] = io_index;
        // TODO [96434]: Support shape infer ops in arbitrary place in pipeline, not just after inputs
        // shape infer ops sequence after input
        const auto& shape_infer_consumers = utils::get_first_child_shape_infer_expr_seq(param);
        for (const auto& child_shape_infer_expr : shape_infer_consumers) {
            manually_assigned_gprs[child_shape_infer_expr->get_output_port_connector(0)] = io_index;
        }
        io_index++;
    }
    for (const auto& result : results) {
        manually_assigned_gprs[result->get_input_port_connector(0)] = io_index;
        // shape infer ops sequence before result
        const auto& shape_infer_sources = utils::get_first_parent_shape_infer_expr_seq(result);
        for (const auto& parent_shape_infer_expr : shape_infer_sources) {
            manually_assigned_gprs[parent_shape_infer_expr->get_input_port_connector(0)] = io_index;
        }
        io_index++;
    }

    size_t counter_vec = 0;
    size_t counter_gpr = 0;
    std::map<tensor, Reg> regs_vec, regs_gpr;
    const auto IS_MANUALLY_ALLOCATED_REG = SIZE_MAX;
    auto accumulator_reg = 0lu;
    for (const auto& expr : exprs) {
        auto op = expr->get_node();
        if (const auto& buffer_expr = ov::as_type_ptr<BufferExpression>(expr)) {
            const auto reg_group = buffer_expr->get_reg_group();
            // All buffers have one common data pointer
            const auto assigned_reg = num_results + num_parameters + reg_group;
            for (const auto& input : expr->get_input_port_connectors()) {
                manually_assigned_gprs[input] = static_cast<Reg>(assigned_reg);
                // shape infer ops in the middle of subgraph. Buffer is inserted before reshape as new loop should start.
                // child shape info ops share the same memory as Buffer.
                const auto& shape_infer_consumers = utils::get_first_child_shape_infer_expr_seq(expr);
                for (const auto& child_shape_infer_expr : shape_infer_consumers) {
                    manually_assigned_gprs[child_shape_infer_expr->get_input_port_connector(0)] =
                        manually_assigned_gprs[child_shape_infer_expr->get_output_port_connector(0)] =
                            static_cast<Reg>(assigned_reg);
                }
            }
            manually_assigned_gprs[expr->get_output_port_connector(0)] = static_cast<Reg>(assigned_reg);
        } else if (ov::is_type<op::HorizonMax>(op) || ov::is_type<op::HorizonSum>(op)) {
            // Only in ReduceDecomposition Reduce ops use HorizonMax/HorizonSum and VectorBuffer.
            // We should manually set the one vector register for VectorBuffer and Max/Sum output to simulate a accumulator
            // TODO [96351]: We should rewrite accumulator pattern using another way
            const auto& input_tensor = expr->get_input_port_connector(0);
            const auto& input_expr = input_tensor->get_source().get_expr();
            const auto& input_expr_input_tensors = input_expr->get_input_port_connectors();
            for (const auto& tensor : input_expr_input_tensors) {
                const auto parent_expr = tensor->get_source().get_expr();
                if (ov::is_type<op::Fill>(parent_expr->get_node())) {
                    if (ov::is_type<op::VectorBuffer>(parent_expr->get_input_port_connector(0)->get_source().get_expr()->get_node())) {
                        manually_assigned_vecs[tensor] = static_cast<Reg>(accumulator_reg);
                        manually_assigned_vecs[parent_expr->get_input_port_connector(0)] = static_cast<Reg>(accumulator_reg);
                    }
                }
            }
            manually_assigned_vecs[input_tensor] = static_cast<Reg>(accumulator_reg);
            accumulator_reg++;
        }
    }
    // Note: have to specify default capture "=" due to MSVC bug (it doesn't capture const expressions implicitly)
    // Otherwise WIN build fails with "IS_MANUALLY_ALLOCATED_REG cannot be implicitly captured because no default capture mode has been specified"
    // the same problem with all the other lambdas in this file
    auto enumerate_out_tensor = [=] (const tensor& out_tensor,
                                     decltype(regs_vec)& reg_map,
                                     const std::map<tensor, Reg>& manually_assigned_regs,
                                     size_t& counter) {
        // Note that some ops might have identical input&output tensors (Result and Tile* for ex.)
        // so we have to check that the tensor has not been enumerated already
        if (reg_map.count(out_tensor) == 0) {
            reg_map[out_tensor] = manually_assigned_regs.count(out_tensor) == 0 ? counter++ : IS_MANUALLY_ALLOCATED_REG;
        }
    };
    for (const auto& expr : exprs) {
        for (size_t i = 0; i < expr->get_output_count(); ++i) {
            const auto& out = expr->get_output_port(i);
            switch (out.get_descriptor_ptr()->get_reg().type) {
                case RegType::vec:
                    enumerate_out_tensor(out.get_port_connector_ptr(), regs_vec, manually_assigned_vecs, counter_vec);
                    break;
                case RegType::gpr:
                    enumerate_out_tensor(out.get_port_connector_ptr(), regs_gpr, manually_assigned_gprs, counter_gpr);
                    break;
                default:
                    OPENVINO_THROW("Unsupported reg type detected");
            }
        }
    }
    // todo: make one for gpr and one for vector
    std::vector<std::set<Reg>> used_gpr, used_vec; // used = used as an input
    std::vector<std::set<Reg>> defined_gpr, defined_vec; // defined = used as output
    used_gpr.reserve(num_expressions);
    used_vec.reserve(num_expressions);
    defined_gpr.reserve(num_expressions);
    defined_vec.reserve(num_expressions);

    auto tensor2reg = [=] (const std::vector<tensor>& tensors, const std::map<tensor, Reg>& reg_map) {
        std::set<Reg> result;
        for (const auto& t : tensors) {
            if (reg_map.count(t) == 0)
                OPENVINO_THROW("Assign registers: attempt to access not enumerated tensor");
            Reg reg_id = reg_map.at(t);
            if (reg_id != IS_MANUALLY_ALLOCATED_REG)
                result.insert(reg_id);
        }
        return result;
    };

    for (const auto& expr : exprs) {
        std::vector<tensor> used_gpr_tensors, used_vec_tensors, defined_gpr_tensors, defined_vec_tensors;
        for (size_t i = 0; i < expr->get_input_count(); ++i) {
            const auto& in = expr->get_input_port(i);
            switch (in.get_descriptor_ptr()->get_reg().type) {
                case RegType::vec:
                    used_vec_tensors.push_back(in.get_port_connector_ptr());
                    break;
                case RegType::gpr:
                    used_gpr_tensors.push_back(in.get_port_connector_ptr());
                    break;
                default:
                    OPENVINO_THROW("Unsupported reg type detected");
            }
        }
        for (size_t i = 0; i < expr->get_output_count(); ++i) {
            const auto& out = expr->get_output_port(i);
            switch (out.get_descriptor_ptr()->get_reg().type) {
                case RegType::vec:
                    defined_vec_tensors.push_back(out.get_port_connector_ptr());
                    break;
                case RegType::gpr:
                    defined_gpr_tensors.push_back(out.get_port_connector_ptr());
                    break;
                default:
                    OPENVINO_THROW("Unsupported reg type detected");
            }
        }
        used_vec.emplace_back(tensor2reg(used_vec_tensors, regs_vec));
        used_gpr.emplace_back(tensor2reg(used_gpr_tensors, regs_gpr));
        defined_vec.emplace_back(tensor2reg(defined_vec_tensors, regs_vec));
        defined_gpr.emplace_back(tensor2reg(defined_gpr_tensors, regs_gpr));
    }

    // define life intervals
    // liveOut[i] - regs that are live on exit from i-th (topologically ordered) operation
    // liveIn[i] - regs that are live on entering the i-th (topologically ordered) operation
    std::vector<std::set<Reg>> life_in_vec(std::move(used_vec)),
                               life_in_gpr(std::move(used_gpr));
    std::vector<std::set<Reg>> life_out_vec(num_expressions, std::set<Reg>()),
                               life_out_gpr(num_expressions, std::set<Reg>());

    // todo: this part if O(N*N), so it's slow for large subgraphs. Can we simplify it? At least add an early stopping criteria
    for (size_t i = 0; i < num_expressions; i++) {
        for (size_t n = 0; n < num_expressions; n++) {
            // Regs that are live on entering the operation = regs used by the op + (all other regs alive - regs defined by the op)
            // copy regs from lifeOut to lifeIn while ignoring regs in def
            std::set_difference(life_out_gpr[n].begin(), life_out_gpr[n].end(),
                                defined_gpr[n].begin(), defined_gpr[n].end(),
                                std::inserter(life_in_gpr[n], life_in_gpr[n].begin()));
            std::set_difference(life_out_vec[n].begin(), life_out_vec[n].end(),
                                defined_vec[n].begin(), defined_vec[n].end(),
                                std::inserter(life_in_vec[n], life_in_vec[n].begin()));
        }
        size_t n = 0;
        for (const auto& expr : exprs) {
            if (is_type<ov::op::v0::Result>(expr->get_node()))
                continue;
            for (const auto& out : expr->get_output_port_connectors()) {
                for (const auto& child_expr_input : out->get_consumers()) {
                    const auto& child_expr = child_expr_input.get_expr();
                    auto child_it = linear_ir.begin();
                    std::advance(child_it, n);
                    size_t k = n;
                    while (child_it != linear_ir.end() && *child_it != child_expr) {
                        child_it++;
                        k++;
                    }
                    if (k == num_expressions)
                        OPENVINO_THROW("assign registers can't find target op in the body");
                    life_out_vec[n].insert(life_in_vec[k].begin(), life_in_vec[k].end());
                    life_out_gpr[n].insert(life_in_gpr[k].begin(), life_in_gpr[k].end());
                }
            }
            n++;
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
    // A variable live interval - is a range (start, stop) of op indexes, such that
    // the variable is alive within this range (defined but not used by the last user)
    std::map<std::pair<int, int>, Reg, by_starting> live_intervals_vec, live_intervals_gpr;

    std::reverse(life_in_vec.begin(), life_in_vec.end());
    std::reverse(life_in_gpr.begin(), life_in_gpr.end());
    auto find_last_use = [](decltype(life_in_gpr) life_in, int i) -> int {
        int ln = static_cast<int>(life_in.size()) - 1;
        for (auto& x : life_in) {
            if (x.find(i) != x.end()) {
                return ln;
            }
            ln--;
        }
        return i;
    };
    for (int i = 0; i < static_cast<int>(num_expressions); i++) {
        for (const auto& def : defined_vec[i])
            live_intervals_vec[std::make_pair(i, find_last_use(life_in_vec, static_cast<int>(def)))] = def;
        for (const auto& def : defined_gpr[i])
            live_intervals_gpr[std::make_pair(i, find_last_use(life_in_gpr, static_cast<int>(def)))] = def;
    }

    auto linescan_assign_registers = [](const decltype(live_intervals_vec)& live_intervals,
                                        const std::set<Reg>& reg_pool) {
        // http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf
        // todo: do we need multimap? <=> can an op have two inputs from the same op?
        std::map<std::pair<int, int>, Reg, by_ending> active;
        // uniquely defined register => reused reg (reduced subset enabled by reg by reusage)
        std::map<Reg, Reg> register_map;
        std::stack<Reg> bank;
        // regs are stored in ascending order in reg_pool, so walk in reverse to assign them the same way
        for (auto rit = reg_pool.crbegin(); rit != reg_pool.crend(); rit++)
            bank.push(*rit);

        std::pair<int, int> interval, active_interval;
        Reg unique_reg, active_unique_reg;
        for (const auto& interval_reg : live_intervals) {
            std::tie(interval, unique_reg) = interval_reg;
            // check expired
            while (!active.empty()) {
                std::tie(active_interval, active_unique_reg) = *active.begin();
                // if end of active interval has not passed yet => stop removing actives since they are sorted by end
                if (active_interval.second >= interval.first) {
                    break;
                }
                active.erase(active_interval);
                bank.push(register_map[active_unique_reg]);
            }
            // allocate
            if (active.size() == reg_pool.size()) {
                // todo: if it is LoopBegin or LoopEnd that requires gpr, and we don't have any in the pool,
                //  then assign SIZE_MAX-1 as a flag to spill a reg inside emitter
                OPENVINO_THROW("can't allocate registers for a snippet ");
            } else {
                register_map[unique_reg] = bank.top();
                bank.pop();
                active.insert(interval_reg);
            }
        }
        return register_map;
    };
    // todo: vec_/gpr_pool are hardware-specific and should be provided by a backend, e.g. overloaded generator
    std::set<Reg> vec_pool;
    for (Reg i = 0; i < reg_count; i++)
        vec_pool.insert(i);
    std::set<Reg> gpr_pool(vec_pool);
    for (const auto& t_reg : manually_assigned_vecs)
        vec_pool.erase(t_reg.second);
    for (const auto& t_reg : manually_assigned_gprs)
        gpr_pool.erase(t_reg.second);
    auto unique2reused_map_vec = linescan_assign_registers(live_intervals_vec, vec_pool);
    auto unique2reused_map_gpr = linescan_assign_registers(live_intervals_gpr, gpr_pool);

    std::map<tensor, Reg> assigned_regs(std::move(manually_assigned_gprs));
    assigned_regs.insert(manually_assigned_vecs.begin(), manually_assigned_vecs.end());
    auto register_assigned_regs = [=, &assigned_regs](const std::map<tensor, Reg>& unique_regs, const std::map<Reg, Reg>& unique2reused) {
        for (const auto& reg : unique_regs) {
            if (reg.second == IS_MANUALLY_ALLOCATED_REG)
                continue;
            if (unique2reused.count(reg.second) == 0)
                OPENVINO_THROW("Assign registers failed to allocate register for a tensor");
            assigned_regs[reg.first] = unique2reused.at(reg.second);
        }
    };
    register_assigned_regs(regs_vec, unique2reused_map_vec);
    register_assigned_regs(regs_gpr, unique2reused_map_gpr);

    for (const auto& expr : exprs) {
        for (size_t i = 0; i < expr->get_input_count(); ++i) {
            expr->get_input_port_descriptor(i)->set_reg_idx(assigned_regs[expr->get_input_port_connector(i)]);
        }
        for (size_t i = 0; i < expr->get_output_count(); ++i) {
            expr->get_output_port_descriptor(i)->set_reg_idx(assigned_regs[expr->get_output_port_connector(i)]);
        }
    }
    return false;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

