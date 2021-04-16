// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_precision.h"

#include <ngraph/pass/manager.hpp>

#include <unordered_set>

namespace MKLDNNPlugin {
namespace pass {

NGRAPH_RTTI_DEFINITION(ConvertPrecision, "MKLDNNConvertPrecision", 0);

namespace {

template <typename K>
struct EnumClassHash {
    std::size_t operator()(K t) const {
        return static_cast<std::size_t>(t);
    }
};

template <typename Key>
using enum_hash_t = typename std::conditional<std::is_enum<Key>::value, EnumClassHash<Key>, std::hash<Key>>::type;
using precisions_set_t = std::unordered_set<ngraph::element::Type_t, enum_hash_t<ngraph::element::Type_t>>;

precisions_set_t find_all_used_precisions(const std::shared_ptr<ngraph::Function> & fn) {
    precisions_set_t used_precisions;

    ngraph::traverse_nodes(fn, [&](std::shared_ptr<ngraph::Node> node) {
        for (auto output : node->outputs()) {
            used_precisions.emplace(output.get_element_type());
        }
        if (auto sub_graph_node = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                auto sub_graph_precisions = find_all_used_precisions(sub_graph);
                used_precisions.insert(sub_graph_precisions.begin(), sub_graph_precisions.end());
            }
        }
    });

    return used_precisions;
}

}   // namespace

const std::vector<std::pair<ngraph::element::Type, ngraph::element::Type>> ConvertPrecision::list = {
    {ngraph::element::i64,     ngraph::element::i32},
    {ngraph::element::u64,     ngraph::element::i32},
    {ngraph::element::i16,     ngraph::element::i32},
    {ngraph::element::u16,     ngraph::element::i32},
    {ngraph::element::u32,     ngraph::element::i32},
    {ngraph::element::f64,     ngraph::element::f32},
    {ngraph::element::f16,     ngraph::element::f32},
    {ngraph::element::boolean, ngraph::element::u8},
    {ngraph::element::i4,      ngraph::element::i8},
    {ngraph::element::u4,      ngraph::element::u8},
};

bool ConvertPrecision::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager(get_pass_config());

    manager.set_per_pass_validation(false);
    auto const used_precisions = find_all_used_precisions(f);
    for (auto &precision : list) {
        if (used_precisions.find(precision.first) != used_precisions.end())
            manager.register_pass<ngraph::pass::ConvertPrecision>(precision.first, precision.second);
    }
    manager.set_per_pass_validation(true);

    manager.run_passes(f);

    // Returning value is false because pass::ConvertPrecision always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass.
    return false;
}

}  // namespace pass
}  // namespace MKLDNNPlugin
