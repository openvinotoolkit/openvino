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
using EqualitySoup = std::shared_ptr<std::set<label_t>>;
using EqTable = std::unordered_map<label_t, EqualitySoup>;
using ValTable = std::unordered_map<label_t, ov::Dimension>;

class OPENVINO_API LabelTable : public std::enable_shared_from_this<LabelTable> {
public:
    explicit LabelTable(label_t label = 1) : current_label(label){};
    void set_as_equal(const ov::Dimension& lhs, const ov::Dimension& rhs);
    bool are_equal(const ov::Dimension& lhs, const ov::Dimension& rhs);

    const EqTable& get_equivalence_table() const;
    const ValTable& get_value_equivalence_table() const;
    label_t get_next_label();

    void set_up_for_tracking(ov::PartialShape& shape);
    void set_up_for_tracking(ov::Dimension& d);
    void set_up_for_tracking(ov::Dimension& d, label_t label);
    static void reset_tracking_info(ov::Dimension& d);

private:
    label_t current_label;
    EqTable dimension_table_of_equivalence;
    ValTable value_table_of_equivalence;
};

}  // namespace ov
