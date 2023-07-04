// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dimension_tracker.hpp"

void ov::TableOfEquivalence::set_as_equal(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    const auto &l_label = ov::DimensionTracker::get_label(lhs), r_label = ov::DimensionTracker::get_label(rhs);
    dimension_table_of_equivalence[l_label].insert(r_label);
    dimension_table_of_equivalence[r_label].insert(l_label);
}

const ov::ValTable& ov::TableOfEquivalence::get_value_equivalence_table() const {
    return value_table_of_equivalence;
}

ov::label_t ov::TableOfEquivalence::get_next_label() {
    return current_label++;
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