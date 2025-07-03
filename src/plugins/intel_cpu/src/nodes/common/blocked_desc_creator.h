// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "cpu_shape.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

class CreatorsMapFilterConstIterator;

class BlockedDescCreator {
public:
    using CreatorPtr = std::shared_ptr<BlockedDescCreator>;
    using CreatorConstPtr = std::shared_ptr<const BlockedDescCreator>;
    using CreatorsMap = std::map<LayoutType, CreatorConstPtr>;
    using Predicate = std::function<bool(const CreatorsMap::value_type&)>;

    static const CreatorsMap& getCommonCreators();
    static std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator> makeFilteredRange(
        const CreatorsMap& map,
        unsigned rank);
    static std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator>
    makeFilteredRange(const CreatorsMap& map, unsigned rank, const std::vector<LayoutType>& supportedTypes);
    static std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator> makeFilteredRange(
        const CreatorsMap& map,
        Predicate predicate);
    [[nodiscard]] virtual CpuBlockedMemoryDesc createDesc(const ov::element::Type& precision,
                                                          const Shape& srcShape) const = 0;

    [[nodiscard]] std::shared_ptr<CpuBlockedMemoryDesc> createSharedDesc(const ov::element::Type& precision,
                                                                         const Shape& srcShape) const {
        return std::make_shared<CpuBlockedMemoryDesc>(createDesc(precision, srcShape));
    }

    [[nodiscard]] virtual size_t getMinimalRank() const = 0;
    virtual ~BlockedDescCreator() = default;
};

class CreatorsMapFilterConstIterator {
public:
    using Iterator = BlockedDescCreator::CreatorsMap::const_iterator;
    using value_type = std::iterator_traits<Iterator>::value_type;
    using reference = std::iterator_traits<Iterator>::reference;
    using pointer = std::iterator_traits<Iterator>::pointer;
    using difference_type = std::iterator_traits<Iterator>::difference_type;
    using iterator_category = std::forward_iterator_tag;
    using predicate_type = std::function<bool(const value_type&)>;

    CreatorsMapFilterConstIterator(predicate_type filter, Iterator begin, Iterator end)
        : _iter(begin),
          _end(end),
          _filter(std::move(filter)) {
        while (_iter != _end && !_filter(*_iter)) {
            ++_iter;
        }
    }
    CreatorsMapFilterConstIterator& operator++() {
        do {
            ++_iter;
        } while (_iter != _end && !_filter(*_iter));
        return *this;
    }

    [[nodiscard]] CreatorsMapFilterConstIterator end() const {
        return {predicate_type(), _end, _end};
    }

    CreatorsMapFilterConstIterator operator++(int) {
        CreatorsMapFilterConstIterator temp(*this);
        ++*this;
        return temp;
    }

    reference operator*() const {
        return *_iter;
    }

    pointer operator->() const {
        return std::addressof(*_iter);
    }

    friend bool operator==(const CreatorsMapFilterConstIterator& lhs, const CreatorsMapFilterConstIterator& rhs) {
        return lhs._iter == rhs._iter;
    }

    friend bool operator!=(const CreatorsMapFilterConstIterator& lhs, const CreatorsMapFilterConstIterator& rhs) {
        return !(lhs == rhs);
    }

private:
    Iterator _iter;
    Iterator _end;
    predicate_type _filter;
};

}  // namespace ov::intel_cpu
