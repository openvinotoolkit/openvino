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
            // todo: optimization: emit bare body with no WA manipulations or pointer increments,
            //  but have to know about tile structure. Communicate from op::Subgraph maybe?
            //if (increment == work_amount)
            const auto work_amount = tile_begin->get_work_amount();
            const auto increment = tile_begin->get_increment();
            // vector tiles are required => Just copy the body, original tile is already a vector one
            if (work_amount >= increment) {
                lower_ops(vector_tile);
            }
            OV_ITT_TASK_NEXT(GENERATE, "::ScalarTile")
            // scalar tiles are required => transform the body into a scalar representation
            if (work_amount % increment != 0) {
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
                // Note: we can use the node map to access scalar counterparts of vector nodes
                // const auto& scalar_tile_begin = ov::as_type_ptr<op::TileBegin>(vector_to_scalar_node_map[tile_begin.get()]);
                const auto& scalar_tile_begin = ov::as_type_ptr<op::TileBegin>(*scalar_tile.begin());
                const auto& scalar_tile_end = ov::as_type_ptr<op::TileBegin>(*scalar_tile.rbegin());
                const auto& finalization_offsets = tile_begin->get_finalization_offsets();
                scalar_tile_begin->set_finalization_offsets(finalization_offsets);
                tile_begin->set_finalization_offsets(std::vector<int64_t>(finalization_offsets.size(), 0));
                scalar_tile_begin->set_work_amount(work_amount % increment);
                // todo: need to communicate to vector tile_end to avoid pop from the stack
                // zero work_amount means that the WA was set in the vector tile and can be reused
//                if (work_amount >= increment)
//                    scalar_tile_begin->set_work_amount(0);
                scalar_tile_begin->set_increment(1);
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