// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/op/tile.hpp"
#include "snippets/op/kernel.hpp"
#include <snippets/itt.hpp>

#include <ngraph/pass/manager.hpp>

auto ngraph::snippets::getRegisters(std::shared_ptr<ngraph::Node>& n) -> ngraph::snippets::RegInfo {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::getRegisters")
    auto rt = n->get_rt_info();

    // ToDo: change to reg_t
    std::vector<size_t> rout;
    auto it_rt = rt.find("reginfo");
    if (it_rt != rt.end()) {
        for (auto reg : it_rt->second.as<std::vector<size_t>>()) {
            rout.push_back(reg);
        }
    }

    std::vector<size_t> rin;
    for (auto input : n->inputs()) {
        auto rt = input.get_source_output().get_node_shared_ptr()->get_rt_info();
        auto it_rt = rt.find("reginfo");
        if (it_rt != rt.end()) {
            for (auto reg : it_rt->second.as<std::vector<size_t>>()) {
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
        throw ngraph_error("unsupported architecture for code genration");

    auto params = m->get_parameters();
    auto results = m->get_results();
    auto in = params.size();
    auto out = results.size();
    auto nptrs = in + out;

    OV_ITT_TASK_CHAIN(GENERATE, ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator", "::VectorTile")
    // vector tile
    std::vector<std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo>> lowered;
    for (auto n : m->get_ordered_ops()) {
        lowered.push_back(std::make_pair(target->get(n->get_type_info())(n), ngraph::snippets::getRegisters(n)));
    }
    OV_ITT_TASK_NEXT(GENERATE, "::ScalarTile")

    // scalar tile
    auto m_scalar = ov::clone_model(*m.get());
    ngraph::pass::Manager mng;
    mng.register_pass<ngraph::snippets::pass::ReplaceLoadsWithScalarLoads>();
    mng.register_pass<ngraph::snippets::pass::ReplaceStoresWithScalarStores>();
    mng.run_passes(m_scalar);
    OV_ITT_TASK_NEXT(GENERATE, "::ScalarTile_get")
    std::vector<std::pair<std::shared_ptr<Emitter>, RegInfo>> scalar_lowered;
    for (auto n : m_scalar->get_ordered_ops()) {
        scalar_lowered.push_back(std::make_pair(target->get(n->get_type_info())(n), ngraph::snippets::getRegisters(n)));
    }
    OV_ITT_TASK_NEXT(GENERATE, "::Tiles1D")

    // wrapping into tiles1D
    std::vector<std::pair<std::shared_ptr<Emitter>, RegInfo>> tiles1D;
    auto tile = std::make_shared<ngraph::snippets::op::Tile>(lowered);
    tile->compile_params = compile_params;
    tiles1D.push_back(std::make_pair(target->get(ngraph::snippets::op::Tile::get_type_info_static())(tile),
                                   std::make_pair(std::vector<size_t>({target->get_lanes(), 0, nptrs, 1}), std::vector<size_t>{})));
    tile = std::make_shared<ngraph::snippets::op::Tile>(scalar_lowered);
    tile->compile_params = compile_params;
    tiles1D.push_back(std::make_pair(target->get(ngraph::snippets::op::Tile::get_type_info_static())(tile),
                    std::make_pair(std::vector<size_t>{{1, target->get_lanes(), nptrs, 1}}, std::vector<size_t>{})));

    OV_ITT_TASK_NEXT(GENERATE, "::Tiles2D")
    // wrapping into tiles2D
    std::vector<std::pair<std::shared_ptr<Emitter>, RegInfo>> tiles2D;
    tile = std::make_shared<ngraph::snippets::op::Tile>(tiles1D);
    tile->compile_params = compile_params;
    tiles2D.push_back(std::make_pair(target->get(ngraph::snippets::op::Tile::get_type_info_static())(tile),
                                     std::make_pair(std::vector<size_t>({1, 0, nptrs, 0}), std::vector<size_t>{})));

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    // emission
    auto tiles2DKernel = std::make_shared<ngraph::snippets::op::Kernel>(tiles2D);
    tiles2DKernel->compile_params = compile_params;
    std::shared_ptr<Emitter> kernel = target->get(ngraph::snippets::op::Kernel::get_type_info_static())(tiles2DKernel);
    kernel->emit_code({in, out}, {});
    OV_ITT_TASK_NEXT(GENERATE, "::EmitData")
    lowered.insert(lowered.end(), scalar_lowered.begin(), scalar_lowered.end());
    for (auto& op : lowered) {
        op.first->emit_data();
    }
    OV_ITT_TASK_NEXT(GENERATE, "::GetSnippet")
    return target->get_snippet();
}
