// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/type.hpp"

#include "ngraph/util.hpp"

namespace {
int compare_types(const ov::DiscreteTypeInfo& v1, const ov::DiscreteTypeInfo& v2) {
    std::string v1_str;
    v1_str += std::to_string(v1.version);
    v1_str += v1.name;
    if (v1.version_id != nullptr) {
        v1_str += v1.version_id;
    } else {
        v1_str += "nullptr";
    }
    std::string v2_str;
    v2_str += std::to_string(v2.version);
    v2_str += v2.name;
    if (v2.version_id != nullptr) {
        v2_str += v2.version_id;
    } else {
        v2_str += "nullptr";
    }
    std::cout << v1_str << " vs " << v2_str << std::endl;
    if (v1_str == v2_str)
        return 0;
    else if (v1_str < v2_str)
        return -1;
    else
        return 1;
}
}  // namespace

namespace std {
size_t std::hash<ngraph::DiscreteTypeInfo>::operator()(const ngraph::DiscreteTypeInfo& k) const {
    NGRAPH_SUPPRESS_DEPRECATED_START
    size_t name_hash = hash<string>()(string(k.name));
    size_t version_hash = hash<decltype(k.version)>()(k.version);
    // don't use parent for hash calculation, it is not a part of type (yet)
    return ngraph::hash_combine(vector<size_t>{name_hash, version_hash});
    NGRAPH_SUPPRESS_DEPRECATED_END
}
}  // namespace std

namespace ov {
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
    return compare_types(*this, b) < 0;
}
bool DiscreteTypeInfo::operator==(const DiscreteTypeInfo& b) const {
    return compare_types(*this, b) == 0;
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
