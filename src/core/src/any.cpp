// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/any.hpp"

#include <array>
#include <limits>
#include <string>
#include <string_view>

#include "openvino/util/common_util.hpp"
namespace {
template <class Container>
bool contains_type_index(Container&& types, const std::type_info& user_type) {
    for (auto&& type : types) {
        if (ov::util::equal(type, user_type)) {
            return true;
        }
    }
    return false;
}
}  // namespace

namespace ov {

bool util::equal(std::type_index lhs, std::type_index rhs) {
    auto result = lhs == rhs;
#if (defined(__ANDROID__) || defined(__APPLE__) || defined(__CHROMIUMOS__)) && defined(__clang__)
    if (!result) {
        result = std::strcmp(lhs.name(), rhs.name()) == 0;
    }
#endif
    return result;
}

Any::Base::~Base() = default;

bool Any::Base::is(const std::type_info& other) const {
    return util::equal(type_info(), other);
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

void Any::Base::read_to(Base& other) const {
    std::stringstream strm;
    print(strm);
    if (other.is<std::string>()) {
        *static_cast<std::string*>(other.addressof()) = strm.str();
    } else {
        if (!strm.str().empty())
            other.read(strm);
    }
}

bool Any::Base::is_base_type_info(const std::type_info& user_type) const {
    return contains_type_index(base_type_info(), user_type);
}

bool Any::Base::is_signed_integral() const {
    return std::is_signed<char>::value ? contains_type_index(std::initializer_list<std::type_index>{typeid(char),
                                                                                                    typeid(signed char),
                                                                                                    typeid(short),
                                                                                                    typeid(int),
                                                                                                    typeid(long),
                                                                                                    typeid(long long)},
                                                             type_info())
                                       : contains_type_index(std::initializer_list<std::type_index>{typeid(signed char),
                                                                                                    typeid(short),
                                                                                                    typeid(int),
                                                                                                    typeid(long),
                                                                                                    typeid(long long)},
                                                             type_info());
}

bool Any::Base::is_unsigned_integral() const {
    return std::is_signed<char>::value
               ? contains_type_index(std::initializer_list<std::type_index>{typeid(unsigned char),
                                                                            typeid(unsigned short),
                                                                            typeid(unsigned int),
                                                                            typeid(unsigned long),
                                                                            typeid(unsigned long long)},
                                     type_info())
               : contains_type_index(std::initializer_list<std::type_index>{typeid(char),
                                                                            typeid(unsigned char),
                                                                            typeid(unsigned short),
                                                                            typeid(unsigned int),
                                                                            typeid(unsigned long),
                                                                            typeid(unsigned long long)},
                                     type_info());
}
bool Any::Base::is_floating_point() const {
    return contains_type_index(
        std::initializer_list<std::type_index>{typeid(float), typeid(double), typeid(long double)},
        type_info());
}

Any::~Any() {
    _temp = {};
    _impl = {};
}

Any::Any(const Any& other) {
    *this = other;
};

Any& Any::operator=(const Any& other) {
    if (other._temp)
        _temp = other._temp->copy();
    if (other._impl)
        _impl = other._impl->copy();
    _so = other._so;
    return *this;
};

Any::Any(const Any& other, const std::shared_ptr<void>& so) : _so{so}, _impl{other._impl} {}

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

void Any::read(std::istream& istream) {
    if (_impl != nullptr) {
        _impl->read(istream);
    }
}

bool Any::operator==(const Any& other) const {
    if (_impl == nullptr || other._impl == nullptr) {
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

void* Any::addressof() {
    return _impl != nullptr ? _impl->addressof() : nullptr;
}

const void* Any::addressof() const {
    return _impl != nullptr ? _impl->addressof() : nullptr;
}
namespace util {

void Read<bool>::operator()(std::istream& is, bool& value) const {
    std::string str;
    is >> str;

    constexpr std::array<std::string_view, 4> off = {"0", "false", "off", "no"};
    constexpr std::array<std::string_view, 4> on = {"1", "true", "on", "yes"};
    str = util::to_lower(str);

    if (std::find(on.begin(), on.end(), str) != on.end()) {
        value = true;
    } else if (std::find(off.begin(), off.end(), str) != off.end()) {
        value = false;
    } else {
        OPENVINO_THROW("Could not convert to bool from string " + str);
    }
}

template <typename F>
static auto stream_to(std::istream& is, F&& f) -> decltype(f(std::declval<const std::string&>())) {
    std::string str;
    is >> str;
    try {
        return f(str);
    } catch (std::exception& e) {
        OPENVINO_THROW(std::string{"Could not convert to: "} +
                       typeid(decltype(f(std::declval<const std::string&>()))).name() + " from string \"" + str +
                       "\": " + e.what());
    }
}

void Read<int>::operator()(std::istream& is, int& value) const {
    value = stream_to(is, [](const std::string& str) {
        return std::stoi(str);
    });
}
void Read<long>::operator()(std::istream& is, long& value) const {
    value = stream_to(is, [](const std::string& str) {
        return std::stol(str);
    });
}
void Read<long long>::operator()(std::istream& is, long long& value) const {
    value = stream_to(is, [](const std::string& str) {
        return std::stoll(str);
    });
}

void Read<unsigned>::operator()(std::istream& is, unsigned& value) const {
    value = stream_to(is, [](const std::string& str) {
        auto ul = std::stoul(str);
        if (ul > std::numeric_limits<unsigned>::max()) {
            throw std::out_of_range{"Out of range"};
        }
        return static_cast<unsigned>(ul);
    });
}
void Read<unsigned long>::operator()(std::istream& is, unsigned long& value) const {
    value = stream_to(is, [](const std::string& str) {
        return std::stoul(str);
    });
}
void Read<unsigned long long>::operator()(std::istream& is, unsigned long long& value) const {
    value = stream_to(is, [](const std::string& str) {
        return std::stoull(str);
    });
}

void Read<float>::operator()(std::istream& is, float& value) const {
    value = stream_to(is, [](const std::string& str) {
        return std::stof(str);
    });
}
void Read<double>::operator()(std::istream& is, double& value) const {
    value = stream_to(is, [](const std::string& str) {
        return std::stod(str);
    });
}
void Read<long double>::operator()(std::istream& is, long double& value) const {
    value = stream_to(is, [](const std::string& str) {
        return std::stold(str);
    });
}

void Read<std::tuple<unsigned int, unsigned int, unsigned int>>::operator()(
    std::istream& is,
    std::tuple<unsigned int, unsigned int, unsigned int>& tuple) const {
    Read<unsigned int>{}(is, std::get<0>(tuple));
    Read<unsigned int>{}(is, std::get<1>(tuple));
    Read<unsigned int>{}(is, std::get<2>(tuple));
}

void Read<AnyMap>::operator()(std::istream& is, AnyMap& map) const {
    char c;

    is >> c;
    OPENVINO_ASSERT(c == '{', "Failed to parse ov::AnyMap. Starting symbols is not '{', it's ", c);

    while (c != '}') {
        std::string key, value;
        std::getline(is, key, ':');
        size_t enclosed_container_level = 0;

        while (is.good()) {
            is >> c;
            if (c == ',') {                         // delimiter between map's pairs
                if (enclosed_container_level == 0)  // we should interrupt after delimiter
                    break;
            }
            if (c == '{' || c == '[')  // case of enclosed maps / arrays
                ++enclosed_container_level;
            if (c == '}' || c == ']') {
                if (enclosed_container_level == 0)
                    break;  // end of map
                --enclosed_container_level;
            }

            value += c;  // accumulate current value
        }
        map.emplace(std::move(key), std::move(value));
    }

    OPENVINO_ASSERT(c == '}', "Failed to parse ov::AnyMap. Ending symbols is not '}', it's ", c);
}

void Read<std::tuple<unsigned int, unsigned int>>::operator()(std::istream& is,
                                                              std::tuple<unsigned int, unsigned int>& tuple) const {
    Read<unsigned int>{}(is, std::get<0>(tuple));
    Read<unsigned int>{}(is, std::get<1>(tuple));
}

void Read<Any>::operator()(std::istream& is, Any& any) const {
    any.read(is);
}

void Write<bool>::operator()(std::ostream& os, const bool& b) const {
    os << (b ? "YES" : "NO");
}

void Write<std::tuple<unsigned int, unsigned int, unsigned int>>::operator()(
    std::ostream& os,
    const std::tuple<unsigned int, unsigned int, unsigned int>& tuple) const {
    os << std::get<0>(tuple) << " " << std::get<1>(tuple) << " " << std::get<2>(tuple);
}

void Write<std::tuple<unsigned int, unsigned int>>::operator()(
    std::ostream& os,
    const std::tuple<unsigned int, unsigned int>& tuple) const {
    os << std::get<0>(tuple) << " " << std::get<1>(tuple);
}

void Write<Any>::operator()(std::ostream& os, const Any& any) const {
    any.print(os);
}

}  // namespace util

template <class U>
[[noreturn]] U Any::Base::convert_impl() const {
    OPENVINO_THROW("Bad cast from: ", type_info().name(), " to: ", typeid(U).name());
}

template <class U, class T, class... Others>
U Any::Base::convert_impl() const {
    return is<T>() ? static_cast<U>(as<T>()) : convert_impl<U, Others...>();
}

template <>
long long Any::Base::convert<long long>() const {
    return std::is_signed<char>::value ? convert_impl<long long, char, signed char, short, int, long, long long>()
                                       : convert_impl<long long, signed char, short, int, long, long long>();
}

template <>
unsigned long long Any::Base::convert<unsigned long long int>() const {
    return std::is_signed<char>::value ? convert_impl<unsigned long long,
                                                      unsigned char,
                                                      unsigned short,
                                                      unsigned int,
                                                      unsigned long,
                                                      unsigned long long>()
                                       : convert_impl<unsigned long long,
                                                      char,
                                                      unsigned char,
                                                      unsigned short,
                                                      unsigned int,
                                                      unsigned long,
                                                      unsigned long long>();
}

template <>
double Any::Base::convert<double>() const {
    return convert_impl<double, float, double>();
}
}  // namespace ov
