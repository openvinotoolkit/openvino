// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
/// \brief Special label value indicate no label set.
constexpr label_t no_label = 0;

/// \brief Friend class of Dimension to set, get and track dimensions and their equivalence
class OPENVINO_API DimensionTracker {
public:
    DimensionTracker() = delete;
    explicit DimensionTracker(const std::shared_ptr<TableOfEquivalence>& table) : m_table_of_equivalence(table) {
        OPENVINO_ASSERT(table != nullptr,
                        "Can not set nullptr as table of equivalence shared pointer for DimensionTracker");
    };

    static void set_label(ov::Dimension& d, label_t label);
    static bool has_label(const ov::Dimension& d);
    static label_t get_label(const ov::Dimension& d);

    void set_table_of_equivalence(ov::Dimension& d) const;
    static const std::shared_ptr<TableOfEquivalence>& get_table_of_equivalence(const ov::Dimension& d);

    void set_up_for_tracking(ov::PartialShape& shape);
    void set_up_for_tracking(ov::Dimension& d);
    void set_up_for_tracking(ov::Dimension& d, label_t label) const;
    static void reset_tracking_info(ov::Dimension& d);

private:
    std::shared_ptr<TableOfEquivalence> m_table_of_equivalence;
};

using EqualitySoup = std::shared_ptr<std::set<label_t>>;
using EqTable = std::unordered_map<label_t, EqualitySoup>;
using ValTable = std::unordered_map<label_t, ov::Dimension>;

class OPENVINO_API TableOfEquivalence {
public:
    explicit TableOfEquivalence(label_t label = 1) : current_label(label){};
    void set_as_equal(const ov::Dimension& lhs, const ov::Dimension& rhs);
    bool are_equal(const ov::Dimension& lhs, const ov::Dimension& rhs);

    const EqTable& get_equivalence_table() const;
    const ValTable& get_value_equivalence_table() const;
    label_t get_next_label();

private:
    label_t current_label;
    EqTable dimension_table_of_equivalence;
    ValTable value_table_of_equivalence;
};

}  // namespace ov
