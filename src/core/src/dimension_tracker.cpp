// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/dimension_tracker.hpp"

using namespace ov;

void TableOfEquivalence::set_as_equal(const Dimension& lhs, const Dimension& rhs) {
    const auto &l_label = DimensionTracker::get_label(lhs), r_label = DimensionTracker::get_label(rhs);
    bool l_known = dimension_table_of_equivalence.count(l_label) && dimension_table_of_equivalence[l_label],
         r_known = dimension_table_of_equivalence.count(r_label) && dimension_table_of_equivalence[r_label];
    if (l_known && r_known) {
        auto soup_l = dimension_table_of_equivalence[l_label];
        soup_l->insert(r_label);
        auto soup_r = dimension_table_of_equivalence[r_label];
        soup_r->insert(l_label);
        soup_l->insert(soup_r->begin(), soup_r->end());
        soup_r->insert(soup_l->begin(), soup_l->end());
    } else {
        auto soup = std::make_shared<std::set<label_t>>();
        if (l_known)
            soup = dimension_table_of_equivalence[l_label];
        else if (r_known)
            soup = dimension_table_of_equivalence[r_label];
        soup->insert(l_label);
        soup->insert(r_label);
        dimension_table_of_equivalence[l_label] = soup;
        dimension_table_of_equivalence[r_label] = soup;
    }
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
    const auto &l_label = DimensionTracker::get_label(lhs), r_label = DimensionTracker::get_label(rhs);
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