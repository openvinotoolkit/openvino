// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/label_table.hpp"

using namespace ov;

void LabelTable::set_as_equal(const Dimension& lhs, const Dimension& rhs) {
    const auto &l_label = lhs.get_label(), r_label = rhs.get_label();
    if (l_label == ov::no_label || r_label == ov::no_label)
        // TODO after value restriction enabling: non labeled dim propagates restriction (if any) to labeled dim
        return;

    auto get_soup = [](const label_t& label, EqTable& table) -> EqualitySoup {
        if (!table.count(label) || !table.at(label))
            table[label] = std::make_shared<std::set<label_t>>(std::set<label_t>{label});
        return table.at(label);
    };

    auto l_soup = get_soup(l_label, dimension_table_of_equivalence);
    auto r_soup = get_soup(r_label, dimension_table_of_equivalence);
    if (r_soup->size() > l_soup->size())  // we would like to minimize number of iterations in the following for-loop
        std::swap(l_soup, r_soup);
    l_soup->insert(r_soup->begin(), r_soup->end());
    for (const auto& label : *r_soup)
        dimension_table_of_equivalence[label] = l_soup;
}

const ValTable& LabelTable::get_value_equivalence_table() const {
    return value_table_of_equivalence;
}

const EqTable& LabelTable::get_equivalence_table() const {
    return dimension_table_of_equivalence;
}

label_t LabelTable::get_next_label() {
    return current_label++;
}

bool LabelTable::are_equal(const Dimension& lhs, const Dimension& rhs) {
    if (!lhs.has_label() || !rhs.has_label())
        return false;
    const auto &l_label = lhs.get_label(), &r_label = rhs.get_label();
    if (l_label == r_label)
        return true;
    if (dimension_table_of_equivalence.count(l_label) && dimension_table_of_equivalence[l_label])
        return dimension_table_of_equivalence[l_label]->count(r_label);
    return false;
}

void LabelTable::reset_tracking_info(Dimension& d) {
    d.set_label(no_label);
    d.set_label_table(nullptr);
}

void LabelTable::set_up_for_tracking(Dimension& d) {
    set_up_for_tracking(d, get_next_label());
}

void LabelTable::set_up_for_tracking(Dimension& d, label_t label) {
    d.set_label(label);  // TODO: should we update current label if user uses larger label?
    d.set_label_table(this->shared_from_this());
}

void LabelTable::set_up_for_tracking(ov::PartialShape& shape) {
    for (auto& d : shape)
        set_up_for_tracking(d);
}
