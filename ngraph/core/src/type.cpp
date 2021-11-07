// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/type.hpp"

#include "ngraph/util.hpp"

namespace std {
size_t std::hash<ngraph::DiscreteTypeInfo>::operator()(const ngraph::DiscreteTypeInfo& k) const {
    return k.hash();
}
}  // namespace std

namespace ov {

size_t DiscreteTypeInfo::hash() const {
    if (hash_value != 0)
        return hash_value;
    size_t name_hash = name ? std::hash<std::string>()(std::string(name)) : 0;
    size_t version_hash = std::hash<decltype(version)>()(version);
    size_t version_id_hash = version_id ? std::hash<std::string>()(std::string(version_id)) : 0;
    // don't use parent for hash calculation, it is not a part of type (yet)
    NGRAPH_SUPPRESS_DEPRECATED_START
    size_t res_value = ngraph::hash_combine(std::vector<size_t>{name_hash, version_hash, version_id_hash});
    NGRAPH_SUPPRESS_DEPRECATED_END
    return res_value;
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
    if (version < b.version)
        return true;
    if (version == b.version && name != nullptr && b.name != nullptr) {
        int cmp_status = strcmp(name, b.name);
        if (cmp_status < 0)
            return true;
        if (cmp_status == 0) {
            std::string v_id(version_id == nullptr ? "" : version_id);
            std::string bv_id(b.version_id == nullptr ? "" : b.version_id);
            if (v_id < bv_id)
                return true;
        }
    }

    return false;
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
