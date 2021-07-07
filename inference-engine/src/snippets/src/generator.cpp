// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"
#include "snippets/register_info.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/op/tile.hpp"
#include "snippets/op/kernel.hpp"

#include <ngraph/pass/manager.hpp>

auto ngraph::snippets::getRegisters(std::shared_ptr<ngraph::Node>& n) -> ngraph::snippets::RegInfo {
    auto rt = n->get_rt_info();

    // ToDo: change to reg_t
    std::vector<size_t> rout;
    if (auto rinfo = rt["reginfo"]) {
        auto reginfo = ngraph::as_type_ptr<ngraph::VariantWrapper<std::vector<size_t>>>(rinfo)->get();
        for (auto reg : reginfo) {
            rout.push_back(reg);
        }
    }

    std::vector<size_t> rin;
    for (auto input : n->inputs()) {
        auto rt = input.get_source_output().get_node_shared_ptr()->get_rt_info();
        if (auto rinfo = rt["reginfo"]) {
            auto reginfo = ngraph::as_type_ptr<ngraph::VariantWrapper<std::vector<size_t>>>(rinfo)->get();
            for (auto reg : reginfo) {
                rin.push_back(reg);
            }
        }
    }
    return std::make_pair(rin, rout);
}

ngraph::snippets::code ngraph::snippets::Generator::generate(std::shared_ptr<ngraph::Function>& f) const {
    if (!target->is_supported())
        throw ngraph_error("unsupported architecture for code genration");

    auto params = f->get_parameters();
    auto results = f->get_results();
    auto nptrs = results.size() + params.size();

    if (nptrs > 7) {
        throw ngraph_error("snippet signature should not exceed 7 arguments. got " + std::to_string(nptrs));
    }

    // vector tile
    std::vector<std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo>> lowered;
    for (auto n : f->get_ordered_ops()) {
        lowered.push_back(std::make_pair(target->get(n->get_type_info())(n), ngraph::snippets::getRegisters(n)));
    }

    // scalar tile
    auto f_scalar = ngraph::clone_function(*f.get());
    ngraph::pass::Manager m;
    m.register_pass<ngraph::snippets::pass::ReplaceLoadsWithScalarLoads>();
    m.register_pass<ngraph::snippets::pass::ReplaceStoresWithScalarStores>();
    m.run_passes(f_scalar);

    std::vector<std::pair<std::shared_ptr<Emitter>, RegInfo>> scalar_lowered;
    for (auto n : f_scalar->get_ordered_ops()) {
        scalar_lowered.push_back(std::make_pair(target->get(n->get_type_info())(n), ngraph::snippets::getRegisters(n)));
    }

    // wrapping into tiles
    std::vector<std::pair<std::shared_ptr<Emitter>, RegInfo>> tiles;
    tiles.push_back(std::make_pair(target->get(ngraph::snippets::op::Tile::type_info)(std::make_shared<ngraph::snippets::op::Tile>(lowered)),
                                   std::make_pair(std::vector<size_t>({target->get_lanes(), nptrs}), std::vector<size_t>{})));
    tiles.push_back(std::make_pair(target->get(ngraph::snippets::op::Tile::type_info)(std::make_shared<ngraph::snippets::op::Tile>(scalar_lowered)),
                    std::make_pair(std::vector<size_t>{{1, nptrs}}, std::vector<size_t>{})));

    // emission
    std::shared_ptr<Emitter> kernel = target->get(ngraph::snippets::op::Kernel::type_info)(std::make_shared<ngraph::snippets::op::Kernel>(tiles));
    kernel->emit_code({params.size(), results.size()}, {});

    lowered.insert(lowered.end(), scalar_lowered.begin(), scalar_lowered.end());
    for (auto& op : lowered) {
        op.first->emit_data();
    }

    return target->get_snippet();
}
