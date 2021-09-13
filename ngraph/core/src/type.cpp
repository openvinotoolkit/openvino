// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/type.hpp"

#include "ngraph/util.hpp"

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
    s << "DiscreteTypeInfo{name: " << info.name << ", version_id: " << info.version_id
      << ", old_version: " << info.version << ", parent: ";
    if (!info.parent)
        s << info.parent;
    else
        s << *info.parent;

    s << "}";
    return s;
}

// parent is commented to fix type relaxed operations
bool DiscreteTypeInfo::operator<(const DiscreteTypeInfo& b) const {
    if (version_id == nullptr || b.version_id == nullptr)
        return version < b.version ||
               (version == b.version && strcmp(name, b.name) < 0);  // ||
                                                                    // (version == b.version && strcmp(name, b.name) ==
                                                                    // 0 && parent && b.parent && *parent < *b.parent);
    else
        return strcmp(version_id, b.version_id) < 0 ||
               (strcmp(version_id, b.version_id) == 0 && strcmp(name, b.name) < 0);  // ||
    // (strcmp(version_id, b.version_id) == 0 && strcmp(name, b.name) == 0 && parent && b.parent &&
    //  *parent < *b.parent);
}
bool DiscreteTypeInfo::operator==(const DiscreteTypeInfo& b) const {
    if (version_id == nullptr || b.version_id == nullptr)
        return version == b.version && strcmp(name, b.name) == 0;  // && parent == b.parent;
    else
        return strcmp(version_id, b.version_id) == 0 && strcmp(name, b.name) == 0;  // && parent == b.parent;
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
