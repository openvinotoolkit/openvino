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

auto getRegisters(std::shared_ptr<ngraph::Node> &n) -> RegInfo {
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
    for (auto n : m->get_ordered_ops()) {
        lowered.emplace_back(std::make_pair(target->get(n->get_type_info())(n), getRegisters(n)));
    }
    OV_ITT_TASK_NEXT(GENERATE, "::ScalarTile")

    // scalar tile
    auto m_scalar = ov::clone_model(*m.get());
    ngraph::pass::Manager mng;
    mng.register_pass<pass::SetScalarCountForLoad>();
    mng.register_pass<pass::SetScalarCountForStore>();
    mng.run_passes(m_scalar);
    OV_ITT_TASK_NEXT(GENERATE, "::ScalarTile_get")
    std::vector<AllocatedEmitter> scalar_lowered;
    for (auto n : m_scalar->get_ordered_ops()) {
        scalar_lowered.emplace_back(std::make_pair(target->get(n->get_type_info())(n), getRegisters(n)));
    }
    OV_ITT_TASK_NEXT(GENERATE, "::Tiles1D");
    // wrapping into tiles1D
    //todo: in, out, and io_last_dims should derive naturally from the graph representation
    const auto& vector_tile = std::make_shared<op::Tile>(lowered, target->get_lanes(), in, out, io_last_dims, io_data_sizes);
    const auto& vector_region = std::make_pair(target->get(op::Tile::get_type_info_static())(vector_tile),
                                               std::make_pair(std::vector<size_t>{}, std::vector<size_t>{}));
    const auto& scalar_tile = std::make_shared<op::Tile>(scalar_lowered, 1, in, out, io_last_dims, io_data_sizes);
    const auto& scalar_region = std::make_pair(target->get(op::Tile::get_type_info_static())(scalar_tile),
                                               std::make_pair(std::vector<size_t>{}, std::vector<size_t>{}));

    OV_ITT_TASK_NEXT(GENERATE, "::Tiles2D")
    // If compile params are provided then it's a static case
    AllocatedEmitter tile_scheduler_region;
    auto tile_scheduler = std::make_shared<op::TileScheduler>(vector_region, scalar_region);
    tile_scheduler->compile_params = compile_params;

    tile_scheduler_region =
            std::make_pair(target->get(op::TileScheduler::get_type_info_static())(tile_scheduler),
                           std::make_pair(std::vector<size_t>({in, out, target->get_lanes()}), std::vector<size_t>{}));

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    // emission
    auto tiles2DKernel = std::make_shared<op::Kernel>(std::vector<AllocatedEmitter>{tile_scheduler_region});
    tiles2DKernel->compile_params = compile_params;
    std::shared_ptr<Emitter> kernel = target->get(op::Kernel::get_type_info_static())(tiles2DKernel);
    kernel->emit_code({in, out}, {});
    OV_ITT_TASK_NEXT(GENERATE, "::EmitData")
    // push back Tiles and TileEmitter to emit data appropriately
    lowered.push_back(tile_scheduler_region);
    lowered.push_back(vector_region);
    lowered.push_back(scalar_region);
    lowered.insert(lowered.end(), scalar_lowered.begin(), scalar_lowered.end());
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