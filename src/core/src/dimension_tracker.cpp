// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/dimension_tracker.hpp"

void ov::TableOfEquivalence::set_as_equal(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    const auto &l_label = ov::DimensionTracker::get_label(lhs), r_label = ov::DimensionTracker::get_label(rhs);
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
        auto soup = std::make_shared<std::set<ov::label_t>>();
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

const ov::ValTable& ov::TableOfEquivalence::get_value_equivalence_table() const {
    return value_table_of_equivalence;
}

ov::label_t ov::TableOfEquivalence::get_next_label() {
    return current_label++;
}

bool ov::TableOfEquivalence::are_equal(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    const auto &l_label = ov::DimensionTracker::get_label(lhs), r_label = ov::DimensionTracker::get_label(rhs);
    if (dimension_table_of_equivalence.count(l_label) && dimension_table_of_equivalence[l_label])
        return dimension_table_of_equivalence[l_label]->count(r_label);
    return false;
}

bool ov::DimensionTracker::has_label(const ov::Dimension& d) {
    return d.m_label != no_label;
}

void ov::DimensionTracker::reset_tracking_info(ov::Dimension& d) {
    d.m_label = no_label;
    d.m_table_of_equivalence = nullptr;
}

void ov::DimensionTracker::set_up_for_tracking(ov::Dimension& d) {
    set_up_for_tracking(d, m_table_of_equivalence->get_next_label());
}