// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type.hpp"

#include "openvino/util/common_util.hpp"

namespace std {
size_t std::hash<ov::DiscreteTypeInfo>::operator()(const ov::DiscreteTypeInfo& k) const {
    return k.hash();
}
}  // namespace std

namespace ov {

size_t DiscreteTypeInfo::hash() const {
    if (hash_value != 0)
        return hash_value;
    size_t name_hash = name ? std::hash<std::string>()(std::string(name)) : 0;
    size_t version_id_hash = version_id ? std::hash<std::string>()(std::string(version_id)) : 0;

    return ov::util::hash_combine(std::vector<size_t>{name_hash, version_id_hash});
}

size_t DiscreteTypeInfo::hash() {
    if (hash_value == 0)
        hash_value = static_cast<const DiscreteTypeInfo*>(this)->hash();
    return hash_value;
}

bool DiscreteTypeInfo::is_castable(const DiscreteTypeInfo& target_type) const {
    return *this == target_type || (parent && parent->is_castable(target_type));
}

std::string DiscreteTypeInfo::get_version() const {
    if (version_id) {
        return std::string(version_id);
    }
    return {};
}

DiscreteTypeInfo::operator std::string() const {
    return std::string(name) + "_" + get_version();
}

std::ostream& operator<<(std::ostream& s, const DiscreteTypeInfo& info) {
    std::string version_id = info.version_id ? info.version_id : "(empty)";
    s << "DiscreteTypeInfo{name: " << info.name << ", version_id: " << version_id << ", parent: ";
    if (!info.parent)
        s << info.parent;
    else
        s << *info.parent;

    s << "}";
    return s;
}

// parent is commented to fix type relaxed operations
bool DiscreteTypeInfo::operator<(const DiscreteTypeInfo& b) const {
    if (name != nullptr && b.name != nullptr) {
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
    if (hash_value != 0 && b.hash_value != 0)
        return hash() == b.hash();
    if (name != nullptr && b.name != nullptr) {
        if (strcmp(name, b.name) == 0) {
            std::string v_id(version_id == nullptr ? "" : version_id);
            std::string bv_id(b.version_id == nullptr ? "" : b.version_id);
            if (v_id == bv_id)
                return true;
        }
    }
    return false;
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
