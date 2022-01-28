// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/type.hpp"
#include <iostream>

#include "openvino/core/except.hpp"
#include "openvino/util/common_util.hpp"

namespace std {
size_t std::hash<ngraph::DiscreteTypeInfo>::operator()(const ngraph::DiscreteTypeInfo& k) const {
    static uint64_t count = 0;
    std::cout << count++ << "AAA " << k.name << " " << k.version_id << " " << k.hash() << std::endl;
    return k.hash();
}
}  // namespace std

namespace ov {
size_t DiscreteTypeInfo::hash() const {
    return hash_value;
}

bool DiscreteTypeInfo::is_castable(const DiscreteTypeInfo& target_type) const {
    return *this == target_type || (parent && parent->is_castable(target_type));
}

std::string DiscreteTypeInfo::get_version() const {
    if (version_id) {
        return std::string(version_id);
    }
    return std::to_string(version);
}

DiscreteTypeInfo::operator std::string() const {
    return std::string(name) + "_" + get_version();
}

std::ostream& operator<<(std::ostream& s, const DiscreteTypeInfo& info) {
    std::string version_id = info.version_id ? info.version_id : "(empty)";
    s << "DiscreteTypeInfo{name: " << info.name << ", version_id: " << version_id << ", old_version: " << info.version
      << ", parent: ";
    if (!info.parent)
        s << info.parent;
    else
        s << *info.parent;

    s << "}";
    return s;
}

// parent is commented to fix type relaxed operations
bool DiscreteTypeInfo::operator<(const DiscreteTypeInfo& b) const {
    if (hash_value != 0 && b.hash_value != 0 && hash_value == b.hash_value)
        return false;
    if (version < b.version)
        return true;
    if (version == b.version && name != nullptr && b.name != nullptr) {
        int cmp_status = (name == b.name) ? 0 : strcmp(name, b.name);
        if (cmp_status < 0)
            return true;
        if (cmp_status == 0) {
            return (!version_id && b.version_id) ? true
                                                 : (version_id && b.version_id && version_id != b.version_id)
                                                       ? strcmp(version_id, b.version_id) < 0
                                                       : false;
        }
    }
    return false;
}
bool DiscreteTypeInfo::operator==(const DiscreteTypeInfo& b) const {
    // Legacy logic
    if (version_id != b.version_id && (version_id == nullptr || b.version_id == nullptr))
        return version == b.version && strcmp(name, b.name) == 0;
    if (hash_value != 0 && b.hash_value != 0)
        return hash_value == b.hash_value;
    throw ov::Exception("Looks strange!");
}
bool DiscreteTypeInfo::operator<=(const DiscreteTypeInfo& b) const {
    return *this == b || *this < b;
}
bool DiscreteTypeInfo::operator>(const DiscreteTypeInfo& b) const {
    return !(*this <= b);
}
bool DiscreteTypeInfo::operator>=(const DiscreteTypeInfo& b) const {
    return !(*this < b);
}
bool DiscreteTypeInfo::operator!=(const DiscreteTypeInfo& b) const {
    return !(*this == b);
}
}  // namespace ov
