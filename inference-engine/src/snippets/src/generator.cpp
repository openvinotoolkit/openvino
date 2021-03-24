// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"
#include "snippets/register_info.hpp"

auto ngraph::snippets::getRegisters(std::shared_ptr<ngraph::Node>& n) -> ngraph::snippets::RegInfo {
    auto rt = n->get_rt_info();

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