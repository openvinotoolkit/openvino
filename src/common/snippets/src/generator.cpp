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

auto tail_transformations(NodeVector& tail, const size_t tail_size, const ngraph::snippets::Generator::GeneratorConfig& config) -> void {
    NodeVector updated_tile;
    auto insertFill = [tail_size](const ov::Input<ov::Node>& input) -> std::shared_ptr<ov::Node> {
        auto copyRegInfo = [](const ov::descriptor::Tensor& from, ov::descriptor::Tensor& to) -> void {
            auto rt = from.get_rt_info();
            auto reginfo = rt.find("reginfo");
            if (reginfo != rt.end()) {
                to.get_rt_info()["reginfo"] = reginfo->second;
            }
        };
        std::shared_ptr<ov::Node> fill = nullptr;
        auto& rt = input.get_rt_info();
        auto fill_rt = rt.find("set_fill");
        if (fill_rt != rt.end()) {
            const auto fill_value = fill_rt->second.as<uint32_t>();
            fill = std::make_shared<ngraph::snippets::op::Fill>(input.get_source_output(), tail_size, fill_value);
            input.get_node()->set_argument(input.get_index(), fill);
            // we should explicitly copy reg info because we insert Fill after assign register
            copyRegInfo(fill->get_input_tensor(0), fill->get_output_tensor(0));
        }
        return fill;
    };

    for (auto& op : tail) {
        // We should fill vector regs by float_min and zero to have
        // correct math calculations for ReduceMax and ReduceSum in scalar case.
        // Note: We find Maximum and Add ops because HorizonMax and HorizonSum are outside Loop,
        //       so they are missed in <tail>
        if (config.m_need_fill_tail_register &&
            (ov::is_type<ov::op::v1::Maximum>(op) ||
             ov::is_type<ov::op::v1::Add>(op))) {
            for (auto i = 0; i < op->inputs().size(); ++i) {
                if (auto fill = insertFill(op->input(i))) {
                    updated_tile.push_back(fill);
                }
            }
        } else if (const auto memory_access = std::dynamic_pointer_cast<ngraph::snippets::op::MemoryAccess>(op)) {
            if (memory_access->get_count() != 1) {
                memory_access->set_count(tail_size);
            }
        }
        updated_tile.push_back(op);
    }

    tail = std::move(updated_tile);
}

ngraph::snippets::code ngraph::snippets::Generator::generate(std::shared_ptr<ov::Model>& m,
                                                             const GeneratorConfig& config,
                                                             const void* compile_params) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")
    if (!target->is_supported())
        throw ngraph_error("unsupported architecture for code generation");

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
                std::vector<int64_t> new_finalization_offsets(loop->get_finalization_offsets());
                const auto& ptr_increments = loop->get_ptr_increments();
                for (auto i = 0; i < new_finalization_offsets.size(); i++) {
                    new_finalization_offsets[i] += ptr_increments[i];
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

                if (config.m_optimize_single_evaluation) {
                    // force ptr increments if there is tail
                    optimize_single_evaluation(vector_loop_end, need_tail);
                }

                lower_ops(vector_loop);
            }
            OV_ITT_TASK_NEXT(GENERATE, "::TailLoop")
            // tail is required => transform the body into a tail representation
            // tail loop is fake loop because for tail we should calculate only
            // finalization offsets which are supported by LoopEnd.
            if (need_tail) {
                NodeMap vector_to_tail_node_map;
                tail_loop = ngraph::clone_nodes(vector_loop,  vector_to_tail_node_map);
                tail_transformations(tail_loop, tail_size, config);
                tail_loop_end = ov::as_type_ptr<op::LoopEnd>(*tail_loop.rbegin());
                tail_loop_end->set_finalization_offsets(tail_finalization_offsets);
                tail_loop_end->set_increment(tail_size);
                // ptr increments were set to the old increment, need to update them in accordance with the new one
                tail_loop_end->update_ptr_increments(static_cast<int64_t>(tail_size));
                tail_loop_end->set_work_amount(tail_size);
                tail_loop_end->has_outer_loop = vector_loop_end->has_outer_loop;

                if (config.m_optimize_single_evaluation) {
                    // tail loop is always executed once
                    optimize_single_evaluation(tail_loop_end);
                }

                lower_ops(tail_loop);
            }
        } else {
            lower_ops({*op});
        }
    }

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    //todo: Kernel need info on i/o data access pattern and data shapes to calculate data offsets
    // pass Params and Results
    // todo: it's probably better to move AllocaledEmitter creation inside Kernel constructor
    //  So Kernel accepts only model ptr and target, and creates AllocatedEmitter inside
    //emission
    auto loops2DKernel = std::make_shared<op::Kernel>(lowered, m);
    loops2DKernel->compile_params = compile_params;
    std::shared_ptr<Emitter> kernel = target->get(op::Kernel::get_type_info_static())(loops2DKernel);

    kernel->emit_code({}, {});

    OV_ITT_TASK_NEXT(GENERATE, "::EmitData")
    for (auto& op : lowered) {
        op.first->emit_data();
    }
    OV_ITT_TASK_NEXT(GENERATE, "::GetSnippet")

    // todo: we save lowered to access compiled brgemm kernels on execution time (normally lowered is destructed by then)
    //  remove this when kernel caching is implemented. Don't forget to make generate const method.
    if (config.m_save_lowered_code)
        lowered_saved = lowered;

    return target->get_snippet();
}

std::shared_ptr<const TargetMachine> Generator::get_target_machine() const {
    return target;
}

}// namespace snippets
}// namespace ngraph
