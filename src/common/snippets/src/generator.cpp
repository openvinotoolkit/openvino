// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/op/kernel.hpp"
#include <snippets/itt.hpp>

#include <ngraph/pass/manager.hpp>
#include <openvino/core/type.hpp>

namespace ngraph {
namespace snippets {

auto getRegisters(const std::shared_ptr<ngraph::Node> &n) -> RegInfo {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::getRegisters")

    // ToDo: change to reg_t
    std::vector<size_t> rin, rout;

    for (const auto& output : n->outputs()) {
        const auto& rt = output.get_tensor_ptr()->get_rt_info();
        auto it_rt = rt.find("reginfo");
        if (it_rt != rt.end())
            rout.push_back(it_rt->second.as<size_t>());
    }

    for (const auto& input : n->inputs()) {
        auto rt = input.get_source_output().get_tensor_ptr()->get_rt_info();
        auto it_rt = rt.find("reginfo");
        if (it_rt != rt.end())
            rin.push_back(it_rt->second.as<size_t>());
    }
    return std::make_pair(rin, rout);
}

ngraph::snippets::code ngraph::snippets::Generator::generate(std::shared_ptr<ov::Model>& m,
                                                             const void* compile_params) const {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")
    if (!target->is_supported())
        throw ngraph_error("unsupported architecture for code generation");

    auto params = m->get_parameters();
    auto results = m->get_results();
    auto in = params.size();
    auto out = results.size();

    std::vector<size_t> io_last_dims(in + out);
    std::vector<size_t> io_data_sizes(in + out);
    std::transform(params.begin(), params.end(), io_last_dims.begin(),
                   [](const std::shared_ptr<Node>& n){
                       auto last_dim = n->get_output_partial_shape(0).rbegin();
                       return last_dim->is_dynamic() ? op::Subgraph::DYNAMIC_DIMENSION
                                                     : last_dim->get_length();
                   });
    std::transform(results.begin(), results.end(), io_last_dims.begin() + in,
                   [](const std::shared_ptr<Node> &n) {
                       auto last_dim = n->get_input_partial_shape(0).rbegin();
                       return last_dim->is_dynamic() ? op::Subgraph::DYNAMIC_DIMENSION
                                                     : last_dim->get_length();
                   });
    std::transform(params.begin(), params.end(), io_data_sizes.begin(),
                   [](const std::shared_ptr<Node>& n){return n->get_element_type().size();});
    std::transform(results.begin(), results.end(), io_data_sizes.begin() + in,
                   [](const std::shared_ptr<Node>& n){return n->get_element_type().size();});

    OV_ITT_TASK_CHAIN(GENERATE, ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator", "::VectorTile")
    // vector loop
    std::vector<AllocatedEmitter> lowered;
    auto lower_ops = [&lowered, this](const NodeVector& ops){
        std::transform(ops.begin(), ops.end(), std::back_inserter(lowered),
                       [this](const std::shared_ptr<Node>& n){
                           return std::make_pair(target->get(n->get_type_info())(n), ngraph::snippets::getRegisters(n));
                       });
    };
    // *1* solo vector/tail loop + empty outer loop
    //      => skip increments (both counter & ptr) : set evaluate_once flag
    // *2* solo vector/tail loop + non-empty outer loop
    //      => skip counter increments but perform ptr increments : set evaluate_once,
    //         and perform pointer increments through finalization offsets
    // *3* vector loop(s) + one tail loop
    //      => vector as usual, tail depends on outer loop, see *1* and *2*
    auto optimize_single_evaluation = [](const std::shared_ptr<op::LoopEnd>& loop, bool force_ptr_increment = false) {
        if (loop->get_work_amount() < 2 * loop->get_increment()) {
            loop->set_evaluate_once(true);
            if (force_ptr_increment || loop->has_outer_loop) {
                const auto increment = loop->get_increment();
                std::vector<int64_t> new_finalization_offsets(loop->get_finalization_offsets());
                const auto& apply_increments = loop->get_apply_increment();
                for (auto i = 0; i < new_finalization_offsets.size(); i++) {
                    new_finalization_offsets[i] += increment * apply_increments[i];
                }
                loop->set_finalization_offsets(new_finalization_offsets);
            }
            return true;
        } else {
            return false;
        }
    };
    const auto& ops = m->get_ordered_ops();
    for (auto op = ops.begin(); op < ops.end(); op++) {
        const auto& loop_begin = ov::as_type_ptr<ngraph::snippets::op::LoopBegin>(*op);

        // ignore outer loops and possible manual scalar loops
        if (loop_begin && loop_begin->get_increment() != 1) {
            OV_ITT_TASK_NEXT(GENERATE, "::VectorLoop")
            NodeVector vector_loop, tail_loop;
            std::shared_ptr<op::LoopEnd> vector_loop_end, tail_loop_end;
            vector_loop_end = loop_begin->get_loop_end();
            tail_loop_end = nullptr;
            while (*op != vector_loop_end)
                vector_loop.push_back(*op++);
            vector_loop.push_back(*op);
            const auto work_amount = vector_loop_end->get_work_amount();
            const auto increment = vector_loop_end->get_increment();
            const auto tail_size = work_amount % increment;
            const auto need_tail = tail_size != 0;
            const auto need_vector_loop = work_amount >= increment;
            // Note, that finalization_offsets could be modified inside optimize_single_evaluation,
            // so need to save them here to cover (evaluate_once vector with non-zero finalization_offsets + tail)
            std::vector<int64_t> tail_finalization_offsets = need_tail ? vector_loop_end->get_finalization_offsets() : std::vector<int64_t> {};
            // vector loops are required => Just copy the body, original loop is already a vector one
            if (need_vector_loop) {
                // Note that finalization offsets should be applied after the last iteration.
                // So if there is a tail, then we should apply offsets after it, but not now.
                if (need_tail)
                    vector_loop_end->set_finalization_offsets(std::vector<int64_t>(tail_finalization_offsets.size(), 0));
                // force ptr increments if there is tail
                optimize_single_evaluation(vector_loop_end, need_tail);
                lower_ops(vector_loop);
            }
            OV_ITT_TASK_NEXT(GENERATE, "::TailLoop")
            // tail is required => transform the body into a tail representation
            // tail loop is fake loop because for tail we should calculate only
            // finalization offsets which are supported by LoopEnd.
            if (need_tail) {
                NodeMap vector_to_tail_node_map;
                tail_loop = ngraph::clone_nodes(vector_loop,  vector_to_tail_node_map);
                std::transform(tail_loop.begin(), tail_loop.end(), tail_loop.begin(),
                               [tail_size](const std::shared_ptr<Node>& n){
                                   const auto& memory_access = std::dynamic_pointer_cast<ngraph::snippets::op::MemoryAccess>(n);
                                   if (memory_access && memory_access->get_count() != 1) {
                                       memory_access->set_count(tail_size);
                                   }
                                   return n;
                               });
                tail_loop_end = ov::as_type_ptr<op::LoopEnd>(*tail_loop.rbegin());
                tail_loop_end->set_finalization_offsets(tail_finalization_offsets);
                tail_loop_end->set_increment(tail_size);
                tail_loop_end->set_work_amount(tail_size);
                tail_loop_end->has_outer_loop = vector_loop_end->has_outer_loop;
                // tail loop is always executed once
                optimize_single_evaluation(tail_loop_end);
                lower_ops(tail_loop);
            }
        } else {
            lower_ops({*op});
        }
    }

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    // emission
    auto loops2DKernel = std::make_shared<op::Kernel>(std::vector<AllocatedEmitter>{lowered});
    loops2DKernel->compile_params = compile_params;
    std::shared_ptr<Emitter> kernel = target->get(op::Kernel::get_type_info_static())(loops2DKernel);

    kernel->emit_code({in, out}, {});

    OV_ITT_TASK_NEXT(GENERATE, "::EmitData")
    for (auto& op : lowered) {
        op.first->emit_data();
    }
    OV_ITT_TASK_NEXT(GENERATE, "::GetSnippet")
    return target->get_snippet();
}

std::shared_ptr<const TargetMachine> Generator::get_target_machine() const {
    return target;
}

}// namespace snippets
}// namespace ngraph