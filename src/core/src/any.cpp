// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/any.hpp"

namespace ov {

void Any::Base::type_check(const std::type_info& type_info_) const {
    OPENVINO_ASSERT(type_info() == type_info_, "Bad cast from: ", type_info().name(), " To type: ", type_info_.name());
}

std::shared_ptr<RuntimeAttribute> Any::Base::as_runtime_attribute() const {
    return {};
}

bool Any::Base::is_copyable() const {
    return true;
}
Any Any::Base::init(const std::shared_ptr<Node>&) {
    return {};
}
Any Any::Base::merge(const std::vector<std::shared_ptr<Node>>&) {
    return {};
}
std::string Any::Base::to_string() {
    std::stringstream strm;
    print(strm);
    return strm.str();
}

std::string Any::Base::to_string() const {
    return const_cast<Any::Base*>(this)->to_string();
}

bool Any::Base::visit_attributes(AttributeVisitor&) {
    return false;
}
bool Any::Base::visit_attributes(AttributeVisitor& visitor) const {
    return const_cast<Any::Base*>(this)->visit_attributes(visitor);
}

Any::~Any() {
    _runtime_attribute_impl = {};
    _impl = {};
}

Any::Any(const Any& other, const std::shared_ptr<void>& so) : _impl{other._impl}, _so{so} {}

Any::Any(const char* str) : Any(std::string{str}) {}

Any::Any(const std::nullptr_t) : Any() {}

void Any::impl_check() const {
    OPENVINO_ASSERT(_impl != nullptr, "Any was not initialized.");
}

const std::type_info& Any::type_info() const {
    impl_check();
    return _impl->type_info();
}

bool Any::empty() const {
    return _impl == nullptr;
}

void Any::print(std::ostream& ostream) const {
    if (_impl != nullptr) {
        _impl->print(ostream);
    }
}

bool Any::operator==(const Any& other) const {
    if (_impl == nullptr && other._impl == nullptr) {
        return false;
    }
    if (_impl == other._impl) {
        return true;
    }
    return _impl->equal(*other._impl);
}

bool Any::operator==(const std::nullptr_t&) const {
    return _impl == nullptr;
}

bool Any::operator!=(const Any& other) const {
    return !operator==(other);
}

Any::Base* Any::operator->() {
    return _impl.get();
}

const Any::Base* Any::operator->() const {
    return _impl.get();
}
}  // namespace ov
