// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/axis_set.hpp"

#include "openvino/util/common_util.hpp"

ov::AxisSet::AxisSet() : std::set<size_t>() {}

ov::AxisSet::AxisSet(const std::initializer_list<size_t>& axes) : std::set<size_t>(axes) {}

ov::AxisSet::AxisSet(const std::set<size_t>& axes) : std::set<size_t>(axes) {}

ov::AxisSet::AxisSet(const std::vector<size_t>& axes) : std::set<size_t>(axes.begin(), axes.end()) {}

ov::AxisSet::AxisSet(const AxisSet& axes) : std::set<size_t>(axes) {}

ov::AxisSet& ov::AxisSet::operator=(const AxisSet& v) {
    static_cast<std::set<size_t>*>(this)->operator=(v);
    return *this;
}

ov::AxisSet& ov::AxisSet::operator=(AxisSet&& v) noexcept {
    static_cast<std::set<size_t>*>(this)->operator=(std::move(v));
    return *this;
}

std::vector<int64_t> ov::AxisSet::to_vector() const {
    return std::vector<int64_t>(this->begin(), this->end());
}

std::ostream& ov::operator<<(std::ostream& s, const AxisSet& axis_set) {
    s << "AxisSet{";
    s << ov::util::join(axis_set);
    s << "}";
    return s;
}

const std::vector<int64_t>& ov::AttributeAdapter<ov::AxisSet>::get() {
    if (!m_buffer_valid) {
        m_buffer.clear();
        for (auto elt : m_ref) {
            m_buffer.push_back(elt);
        }
        m_buffer_valid = true;
    }
    return m_buffer;
}

void ov::AttributeAdapter<ov::AxisSet>::set(const std::vector<int64_t>& value) {
    m_ref = ov::AxisSet();
    for (auto elt : value) {
        m_ref.insert(elt);
    }
    m_buffer_valid = false;
}
