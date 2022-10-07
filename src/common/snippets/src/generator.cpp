// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/op/tile.hpp"
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
        if (it_rt != rt.end()) {
            for (auto reg : it_rt->second.as<std::vector<size_t>>()) {
                rout.push_back(reg);
            }
        }
    }

    for (const auto& input : n->inputs()) {
        auto rt = input.get_source_output().get_tensor_ptr()->get_rt_info();
        auto it_rt = rt.find("reginfo");
        if (it_rt != rt.end()) {
            for (auto& reg : it_rt->second.as<std::vector<size_t>>()) {
                rin.push_back(reg);
            }
        }
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
    // vector tile
    std::vector<AllocatedEmitter> lowered;
    auto lower_ops = [&lowered, this](const NodeVector& ops){
        std::transform(ops.begin(), ops.end(), std::back_inserter(lowered),
                       [this](const std::shared_ptr<Node>& n){
                           return std::make_pair(target->get(n->get_type_info())(n), ngraph::snippets::getRegisters(n));
                       });
    };
    // *1* solo vector/scalar tile + empty outer tile
    //      => skip increments (both counter & ptr) : set evaluate_once flag
    // *2* solo vector/scalar tile + non-empty outer tile
    //      => skip counter increments but perform ptr increments : set evaluate_once,
    //         and perform pointer increments through finalization offsets
    // *3* one vector tile + multiple scalar tiles
    //      => vector force_ptr_increment=true to enable *2*, scalar as usual
    // *4* vector tile(s) + one scalar tile
    //      => vector as usual, scalar depends on outer tile, see *1* and *2*
    auto optimize_single_evaluation = [](const std::shared_ptr<op::TileEnd>& tile, bool force_ptr_increment = false) {
        if (tile->get_work_amount() < 2 * tile->get_increment()) {
            tile->set_evaluate_once(true);
            if (force_ptr_increment || tile->has_outer_tile) {
                const auto increment = tile->get_increment();
                std::vector<int64_t> new_finalization_offsets(tile->get_finalization_offsets());
                const auto& apply_increments = tile->get_apply_increment();
                for (auto i = 0; i < new_finalization_offsets.size(); i++) {
                    new_finalization_offsets[i] += increment * apply_increments[i];
                }
                tile->set_finalization_offsets(new_finalization_offsets);
            }
            return true;
        } else {
            return false;
        }
    };
    const auto& ops = m->get_ordered_ops();
    for (auto op = ops.begin(); op < ops.end(); op++) {
        const auto& tile_begin = ov::as_type_ptr<ngraph::snippets::op::TileBegin>(*op);
        // ignore outer tiles and possible manual scalar tiles
        if (tile_begin && tile_begin->get_increment() != 1) {
            NodeVector vector_tile;
            const auto& tile_end = tile_begin->get_tile_end();
            while (*op != tile_end)
                vector_tile.push_back(*op++);
            vector_tile.push_back(*op);
            const auto work_amount = tile_end->get_work_amount();
            const auto increment = tile_end->get_increment();
            const auto need_scalar_tile = work_amount % increment != 0;
            const auto need_vector_tile = work_amount >= increment;
            std::vector<int64_t> scalar_finalization_offsets = need_scalar_tile ? tile_end->get_finalization_offsets()
                                                                         : std::vector<int64_t> {};
            bool vector_evaluate_once = false;
            // vector tiles are required => Just copy the body, original tile is already a vector one
            if (need_vector_tile) {
                // Note that finalization offsets should be applied after the last iteration.
                // So if there is a scalar tile, then we should apply offsets after it, but not now.
                if (need_scalar_tile) {
                    tile_end->set_finalization_offsets(std::vector<int64_t>(scalar_finalization_offsets.size(), 0));
                }
                // force ptr increments if there is at least one scalar tile
                vector_evaluate_once = optimize_single_evaluation(tile_end, need_scalar_tile);
                // can't let scalar tile to reuse reg_work_amount, since it's not set if vector_evaluate_once==true
                tile_end->reuse_work_amount_reg = !vector_evaluate_once && need_scalar_tile;
                lower_ops(vector_tile);
            }
            OV_ITT_TASK_NEXT(GENERATE, "::ScalarTile")
            // scalar tiles are required => transform the body into a scalar representation
            if (need_scalar_tile) {
                NodeMap vector_to_scalar_node_map;
                NodeVector scalar_tile = ngraph::clone_nodes(vector_tile,  vector_to_scalar_node_map);
                std::transform(scalar_tile.begin(), scalar_tile.end(), scalar_tile.begin(),
                               [](const std::shared_ptr<Node>& n){
                                   if (const auto load = ov::as_type_ptr<ngraph::snippets::op::Load>(n))
                                       load->set_count(1);
                                   else if (const auto store = ov::as_type_ptr<ngraph::snippets::op::Store>(n))
                                       store->set_count(1);
                                   return n;
                               });
                const auto& scalar_tile_end = ov::as_type_ptr<op::TileEnd>(*scalar_tile.rbegin());
                scalar_tile_end->set_finalization_offsets(scalar_finalization_offsets);
                const auto scalar_work_amount = work_amount % increment;
                scalar_tile_end->set_increment(1);
                scalar_tile_end->set_work_amount(scalar_work_amount);
                scalar_tile_end->has_outer_tile = tile_end->has_outer_tile;
                // ptr increment is applied automatically if there is non-empty outer tile
                optimize_single_evaluation(scalar_tile_end);
                if (need_vector_tile && !vector_evaluate_once) {
                    scalar_tile_end->get_tile_begin()->reuse_work_amount_reg = true;
                }
                lower_ops(scalar_tile);
            }
        } else {
            lower_ops({*op});
        }
    }

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    // emission
    auto tiles2DKernel = std::make_shared<op::Kernel>(std::vector<AllocatedEmitter>{lowered});
    tiles2DKernel->compile_params = compile_params;
    std::shared_ptr<Emitter> kernel = target->get(op::Kernel::get_type_info_static())(tiles2DKernel);

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