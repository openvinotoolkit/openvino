// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/type.hpp"

#include "ngraph/util.hpp"

namespace std {
size_t std::hash<ngraph::DiscreteTypeInfo>::operator()(const ngraph::DiscreteTypeInfo& k) const {
    if (k.hash_value != 0)
        return k.hash_value;
    NGRAPH_SUPPRESS_DEPRECATED_START
    size_t name_hash = k.name ? hash<string>()(string(k.name)) : 0;
    size_t version_hash = hash<decltype(k.version)>()(k.version);
    size_t version_id_hash = k.version_id ? hash<string>()(string(k.version_id)) : 0;
    // don't use parent for hash calculation, it is not a part of type (yet)
    return ngraph::hash_combine(vector<size_t>{name_hash, version_hash, version_id_hash});
    NGRAPH_SUPPRESS_DEPRECATED_END
}
}  // namespace std

namespace ov {

DiscreteTypeInfo::DiscreteTypeInfo(const char* _name,
                                   uint64_t _version,
                                   const char* _version_id,
                                   const DiscreteTypeInfo* _parent)
    : name(_name),
      version(_version),
      version_id(_version_id),
      parent(_parent),
      hash_value(0) {
    hash_value = std::hash<ov::DiscreteTypeInfo>{}(*this);
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
    if (hash_value == 0 || b.hash_value == 0)
        return version < b.version || (version == b.version && strcmp(name, b.name) < 0);
    bool less = version < b.version;
    int cmp_status = 0;
    if (!less && version == b.version && name != nullptr && b.name != nullptr) {
        cmp_status = strcmp(name, b.name);
        less |= cmp_status < 0;
    }
    if (!less && cmp_status == 0 && version_id != nullptr && b.version_id != nullptr) {
        cmp_status = strcmp(version_id, b.version_id);
        less |= cmp_status < 0;
    }

    return less;
}
bool DiscreteTypeInfo::operator==(const DiscreteTypeInfo& b) const {
    if (hash_value == 0 || b.hash_value == 0)
        return version == b.version && strcmp(name, b.name) == 0;  // && parent == b.parent;
    return hash_value == b.hash_value;
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
