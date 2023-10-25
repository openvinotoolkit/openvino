// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/dimension_tracker.hpp"

using namespace ov;

void TableOfEquivalence::set_as_equal(const Dimension& lhs, const Dimension& rhs) {
    const auto &l_label = DimensionTracker::get_label(lhs), r_label = DimensionTracker::get_label(rhs);
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

const ValTable& TableOfEquivalence::get_value_equivalence_table() const {
    return value_table_of_equivalence;
}

const EqTable& TableOfEquivalence::get_equivalence_table() const {
    return dimension_table_of_equivalence;
}

label_t TableOfEquivalence::get_next_label() {
    return current_label++;
}

bool TableOfEquivalence::are_equal(const Dimension& lhs, const Dimension& rhs) {
    if (!DimensionTracker::has_label(lhs) || !DimensionTracker::has_label(rhs))
        return false;
    const auto &l_label = DimensionTracker::get_label(lhs), &r_label = DimensionTracker::get_label(rhs);
    if (l_label == r_label)
        return true;
    if (dimension_table_of_equivalence.count(l_label) && dimension_table_of_equivalence[l_label])
        return dimension_table_of_equivalence[l_label]->count(r_label);
    return false;
}

void DimensionTracker::set_label(Dimension& d, label_t label) {
    OPENVINO_ASSERT(label != no_label, "Can not set zero as label for dimension -- it is reserved for no label");
    d.m_label = label;
}

bool DimensionTracker::has_label(const Dimension& d) {
    return d.m_label != no_label;
}

label_t DimensionTracker::get_label(const Dimension& d) {
    return d.m_label;
}

const std::shared_ptr<TableOfEquivalence>& DimensionTracker::get_table_of_equivalence(const Dimension& d) {
    return d.m_table_of_equivalence;
}

void DimensionTracker::set_table_of_equivalence(Dimension& d) const {
    OPENVINO_ASSERT(d.m_table_of_equivalence == nullptr, "Dimension is already being tracked");
    OPENVINO_ASSERT(m_table_of_equivalence != nullptr, "Can not set nullptr as table of equivalence shared pointer");
    d.m_table_of_equivalence = m_table_of_equivalence;
}

void DimensionTracker::reset_tracking_info(Dimension& d) {
    d.m_label = no_label;
    d.m_table_of_equivalence = nullptr;
}

void DimensionTracker::set_up_for_tracking(Dimension& d) {
    set_up_for_tracking(d, m_table_of_equivalence->get_next_label());
}

void DimensionTracker::set_up_for_tracking(Dimension& d, label_t label) const {
    set_label(d, label);
    set_table_of_equivalence(d);
}

void DimensionTracker::set_up_for_tracking(ov::PartialShape& shape) {
    for (auto& d : shape)
        set_up_for_tracking(d);
}
