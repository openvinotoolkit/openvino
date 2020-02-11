// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include <vpu/utils/range/helpers.hpp>
#include <vpu/utils/checked_cast.hpp>

namespace vpu {

namespace details {

template <class Container>
auto getFrontImpl_(const Container& cont, int) -> decltype(cont.front()) {
    assert(!cont.empty());
    return cont.front();
}
template <class Container>
auto getFrontImpl_(const Container& cont, ...) -> decltype(*cont.begin()) {
    assert(!cont.empty());
    return *cont.begin();
}
template <class Container>
auto getFrontImpl(const Container& cont) -> decltype(getFrontImpl_(cont, 0)) {
    return getFrontImpl_(cont, 0);
}

template <class Container>
auto getBackImpl_(const Container& cont, int) -> decltype(cont.back()) {
    assert(!cont.empty());
    return cont.back();
}
template <class Container>
auto getBackImpl_(const Container& cont, ...) -> decltype(*cont.rbegin()) {
    assert(!cont.empty());
    return *cont.rbegin();
}
template <class Container>
auto getBackImpl(const Container& cont) -> decltype(getBackImpl_(cont, 0)) {
    return getBackImpl_(cont, 0);
}

template <class Container, bool makeCopy>
class ContainerRange final {
public:
    static constexpr bool has_reverse_iter = HasReverseIterator<Container>::value;
    static constexpr bool has_random_access = HasRandomAccess<Container>::value;
    static constexpr bool const_time_size = true;

    using value_type = typename Container::value_type;
    using size_type = std::size_t;

    using iterator = typename Container::const_iterator;
    using reverse_iterator = typename GetReverseIterator<Container, has_reverse_iter>::reverse_iterator;

    using const_iterator = iterator;
    using const_reverse_iterator = reverse_iterator;

    ContainerRange() = default;
    explicit ContainerRange(Container cont) : _cont(std::move(cont)) {}

    const_iterator begin() const {
        return _cont.begin();
    }
    const_iterator end() const {
        return _cont.end();
    }

    template <typename Q = Container, typename = typename std::enable_if<HasReverseIterator<Q>::value>::type>
    const_reverse_iterator rbegin() const {
        return _cont.rbegin();
    }
    template <typename Q = Container, typename = typename std::enable_if<HasReverseIterator<Q>::value>::type>
    const_reverse_iterator rend() const {
        return _cont.rend();
    }

    auto front() const ->
            decltype(getFrontImpl(std::declval<const Container&>())) {
        return getFrontImpl(_cont);
    }
    template <typename Q = Container, typename = typename std::enable_if<HasReverseIterator<Q>::value>::type>
    auto back() const ->
            decltype(getBackImpl(std::declval<const Container&>())) {
        return getBackImpl(_cont);
    }

    size_type size() const {
        return checked_cast<size_type>(_cont.size());
    }

    bool empty() const {
        return _cont.empty();
    }

    template <typename Q = Container, typename = typename std::enable_if<HasRandomAccess<Q>::value>::type>
    auto operator[](int ind) const ->
            decltype(std::declval<const Q&>().operator[](ind)) {
        assert(ind >= 0 && ind < _cont.size());
        return _cont[ind];
    }

private:
    Container _cont;
};

template <class Container>
class ContainerRange<Container, false> final {
public:
    static constexpr bool has_reverse_iter = HasReverseIterator<Container>::value;
    static constexpr bool has_random_access = HasRandomAccess<Container>::value;
    static constexpr bool const_time_size = true;

    using value_type = typename Container::value_type;
    using size_type = std::size_t;

    using iterator = typename Container::const_iterator;
    using reverse_iterator = typename GetReverseIterator<Container, has_reverse_iter>::reverse_iterator;

    using const_iterator = iterator;
    using const_reverse_iterator = reverse_iterator;

    ContainerRange() = default;
    explicit ContainerRange(const Container& cont) : _cont(&cont) {}

    const_iterator begin() const {
        assert(_cont != nullptr);
        return _cont->begin();
    }
    const_iterator end() const {
        assert(_cont != nullptr);
        return _cont->end();
    }

    template <typename Q = Container, typename = typename std::enable_if<HasReverseIterator<Q>::value>::type>
    const_reverse_iterator rbegin() const {
        assert(_cont != nullptr);
        return _cont->rbegin();
    }
    template <typename Q = Container, typename = typename std::enable_if<HasReverseIterator<Q>::value>::type>
    const_reverse_iterator rend() const {
        assert(_cont != nullptr);
        return _cont->rend();
    }

    auto front() const ->
            decltype(getFrontImpl(std::declval<const Container&>())) {
        assert(_cont != nullptr);
        return getFrontImpl(*_cont);
    }
    template <typename Q = Container, typename = typename std::enable_if<HasReverseIterator<Q>::value>::type>
    auto back() const ->
            decltype(getBackImpl(std::declval<const Container&>())) {
        assert(_cont != nullptr);
        return getBackImpl(*_cont);
    }

    size_type size() const {
        assert(_cont != nullptr);
        return checked_cast<size_type>(_cont->size());
    }

    bool empty() const {
        assert(_cont != nullptr);
        return _cont->empty();
    }

    template <typename Q = Container, typename = typename std::enable_if<HasRandomAccess<Q>::value>::type>
    auto operator[](int ind) const ->
            decltype(std::declval<const Q&>().operator[](ind)) {
        assert(_cont != nullptr);
        assert(ind >= 0 && ind < _cont->size());
        return (*_cont)[ind];
    }

private:
    const Container* _cont = nullptr;
};

template <class Container>
ContainerRange<typename std::decay<Container>::type, true> containerRange(Container&& cont, std::true_type) {
    return ContainerRange<typename std::decay<Container>::type, true>(std::forward<Container>(cont));
}
template <class Container>
ContainerRange<typename std::decay<Container>::type, false> containerRange(const Container& cont, std::false_type) {
    return ContainerRange<typename std::decay<Container>::type, false>(cont);
}

}  // namespace details

template <class Container>
auto containerRange(Container&& cont) ->
        decltype(details::containerRange(std::forward<Container>(cont), std::is_rvalue_reference<Container&&>{})) {
    return details::containerRange(std::forward<Container>(cont), std::is_rvalue_reference<Container&&>{});
}

namespace details {

struct ContainerRangeTag final {};

template <class Container>
auto operator|(Container&& cont, ContainerRangeTag&&) ->
        decltype(vpu::containerRange(std::forward<Container>(cont))) {
    return vpu::containerRange(std::forward<Container>(cont));
}

}  // namespace details

inline details::ContainerRangeTag asRange() {
    return details::ContainerRangeTag{};
}

}  // namespace vpu
