// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/any.hpp"

#include <limits>
#include <string>

namespace ov {

bool Any::equal(std::type_index lhs, std::type_index rhs) {
    auto result = lhs == rhs;
#if (defined(__ANDROID__) || defined(__APPLE__)) && defined(__clang__)
    if (!result) {
        result = std::strcmp(lhs.name(), rhs.name()) == 0;
    }
#endif
    return result;
}

bool Any::Base::is(const std::type_info& other) const {
    return Any::equal(type_info(), other);
}

void Any::Base::type_check(const std::type_info& type_info_) const {
    OPENVINO_ASSERT(is(type_info_), "Bad cast from: ", type_info().name(), " To type: ", type_info_.name());
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
    _temp_impl = {};
    _impl = {};
}

Any::Any(const Any& other, const std::shared_ptr<void>& so) : _impl{other._impl}, _so{so} {}

Any::Any(const char* str) : Any(std::string{str}) {}

Any::Any(const std::nullptr_t) : Any() {}

void Any::impl_check() const {
    if (_impl == nullptr) {
        OPENVINO_UNREACHABLE("Any was not initialized.");
    }
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

void Any::read(std::istream& istream) {
    if (_impl != nullptr) {
        _impl->read(istream);
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

void Any::read_impl(std::istream& is, bool& value) {
    std::string str;
    is >> str;
    if (str == "YES") {
        value = true;
    } else if (str == "NO") {
        value = false;
    } else {
        OPENVINO_UNREACHABLE("Could not convert to bool from string " + str);
    }
}

template <typename F>
static auto stream_to(std::istream& is, F&& f) -> decltype(f(std::declval<const std::string&>())) {
    std::string str;
    is >> str;
    try {
        return f(str);
    } catch (std::exception& e) {
        OPENVINO_UNREACHABLE(std::string{"Could not convert to: "} +
                             typeid(decltype(f(std::declval<const std::string&>()))).name() + " from string " + str +
                             ": " + e.what());
    }
}

void Any::read_impl(std::istream& is, int& value) {
    value = stream_to(is, [](const std::string& str) {
        return std::stoi(str);
    });
}
void Any::read_impl(std::istream& is, long& value) {
    value = stream_to(is, [](const std::string& str) {
        return std::stol(str);
    });
}
void Any::read_impl(std::istream& is, long long& value) {
    value = stream_to(is, [](const std::string& str) {
        return std::stoll(str);
    });
}

void Any::read_impl(std::istream& is, unsigned& value) {
    value = stream_to(is, [](const std::string& str) {
        auto ul = std::stoul(str);
        if (ul > std::numeric_limits<unsigned>::max()) {
            throw std::out_of_range{"Out of range"};
        }
        return static_cast<unsigned>(ul);
    });
}
void Any::read_impl(std::istream& is, unsigned long& value) {
    value = stream_to(is, [](const std::string& str) {
        return std::stoul(str);
    });
}
void Any::read_impl(std::istream& is, unsigned long long& value) {
    value = stream_to(is, [](const std::string& str) {
        return std::stoull(str);
    });
}

void Any::read_impl(std::istream& is, float& value) {
    value = stream_to(is, [](const std::string& str) {
        return std::stof(str);
    });
}
void Any::read_impl(std::istream& is, double& value) {
    value = stream_to(is, [](const std::string& str) {
        return std::stod(str);
    });
}
void Any::read_impl(std::istream& is, long double& value) {
    value = stream_to(is, [](const std::string& str) {
        return std::stold(str);
    });
}

void Any::print_impl(std::ostream& os, const bool& b) {
    os << (b ? "YES" : "NO");
}
}  // namespace ov
