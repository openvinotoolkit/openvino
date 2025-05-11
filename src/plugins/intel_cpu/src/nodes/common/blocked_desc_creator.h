// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "cpu_shape.h"
#include "memory_desc/cpu_blocked_memory_desc.h"

namespace ov {
namespace intel_cpu {

class CreatorsMapFilterConstIterator;

class BlockedDescCreator {
public:
    typedef std::shared_ptr<BlockedDescCreator> CreatorPtr;
    typedef std::shared_ptr<const BlockedDescCreator> CreatorConstPtr;
    typedef std::map<LayoutType, CreatorConstPtr> CreatorsMap;
    typedef std::function<bool(const CreatorsMap::value_type&)> Predicate;

public:
    static const CreatorsMap& getCommonCreators();
    static std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator> makeFilteredRange(
        const CreatorsMap& map,
        unsigned rank);
    static std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator>
    makeFilteredRange(const CreatorsMap& map, unsigned rank, const std::vector<LayoutType>& supportedTypes);
    static std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator> makeFilteredRange(
        const CreatorsMap& map,
        Predicate predicate);
    virtual CpuBlockedMemoryDesc createDesc(const ov::element::Type& precision, const Shape& srcShape) const = 0;

    std::shared_ptr<CpuBlockedMemoryDesc> createSharedDesc(const ov::element::Type& precision,
                                                           const Shape& srcShape) const {
        return std::make_shared<CpuBlockedMemoryDesc>(createDesc(precision, srcShape));
    }

    virtual size_t getMinimalRank() const = 0;
    virtual ~BlockedDescCreator() = default;
};

class CreatorsMapFilterConstIterator {
public:
    typedef BlockedDescCreator::CreatorsMap::const_iterator Iterator;
    typedef std::iterator_traits<Iterator>::value_type value_type;
    typedef std::iterator_traits<Iterator>::reference reference;
    typedef std::iterator_traits<Iterator>::pointer pointer;
    typedef std::iterator_traits<Iterator>::difference_type difference_type;
    typedef std::forward_iterator_tag iterator_category;
    typedef std::function<bool(const value_type&)> predicate_type;

public:
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

    CreatorsMapFilterConstIterator end() const {
        return CreatorsMapFilterConstIterator(predicate_type(), _end, _end);
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

}  // namespace intel_cpu
}  // namespace ov
