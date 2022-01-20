// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "openvino/core/dimension.hpp"

namespace ov {

/// \brief Friend class of Dimension to set, get and track dimensions and their equivalence
class DimensionTracker {
public:
    DimensionTracker() = delete;
    explicit DimensionTracker(const std::shared_ptr<TableOfEquivalence>& table) : m_table_of_equivalence(table) {
        OPENVINO_ASSERT(table != nullptr,
                        "Can not set nullptr as table of equivalence shared pointer for DimensionTracker");
    };

    static void set_label(ov::Dimension& d, size_t label) {
        OPENVINO_ASSERT(label != 0, "Can not set zero as label for dimension -- it is reserved for no label");
        d.m_label = label;
    }

    static size_t get_label(const ov::Dimension& d) {
        return d.m_label;
    }

    void set_table_of_equivalence(ov::Dimension& d) const {
        OPENVINO_ASSERT(d.m_table_of_equivalence == nullptr, "ov::Dimension is already being tracked");
        OPENVINO_ASSERT(m_table_of_equivalence != nullptr,
                        "Can not set nullptr as table of equivalence shared pointer");
        d.m_table_of_equivalence = m_table_of_equivalence;
    }

    const std::shared_ptr<TableOfEquivalence>& get_table_of_equivalence(ov::Dimension& d) const {
        return m_table_of_equivalence;
    }

    void set_up_for_tracking(ov::Dimension& d, size_t label) const {
        set_label(d, label);
        set_table_of_equivalence(d);
    }

    static void reset_tracking_info(ov::Dimension& d) {
        d.m_label = 0;
        d.m_table_of_equivalence = nullptr;
    }

private:
    std::shared_ptr<TableOfEquivalence> m_table_of_equivalence;
};

using EqTable = std::unordered_map<size_t, std::unordered_set<size_t>>;
using ValTable = std::unordered_map<size_t, ov::Dimension>;

class TableOfEquivalence {
public:
    void set_as_equal(const ov::Dimension& lhs, const ov::Dimension& rhs) {
        const auto &l_label = ov::DimensionTracker::get_label(lhs), r_label = ov::DimensionTracker::get_label(rhs);
        dimension_table_of_equivalence[l_label].insert(r_label);
        dimension_table_of_equivalence[r_label].insert(l_label);
    }

    const EqTable& get_equivalence_table() const {
        return dimension_table_of_equivalence;
    }

    const ValTable& get_value_equivalence_table() const {
        return value_table_of_equivalence;
    }

private:
    EqTable dimension_table_of_equivalence;
    ValTable value_table_of_equivalence;
};

}  // namespace ov
