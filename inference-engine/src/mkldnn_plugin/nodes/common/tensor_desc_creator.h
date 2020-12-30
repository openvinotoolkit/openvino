// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layouts.h>
#include <set>

namespace MKLDNNPlugin {

enum class TensorDescCreatorTypes {
    nspc,
    nCsp8c,
    nCsp16c,
    ncsp
};

class CreatorsMapFilterIterator;

class TensorDescCreator {
public:
    typedef std::shared_ptr<TensorDescCreator> CreatorPtr;
    typedef std::shared_ptr<const TensorDescCreator> CreatorConstPtr;
    typedef std::map<TensorDescCreatorTypes, CreatorConstPtr> CreatorsMap;

public:
    static CreatorsMap getCommonCreators();
    static std::pair<CreatorsMapFilterIterator, CreatorsMapFilterIterator>
    makeFilteredRange(CreatorsMap& map, unsigned rank);
    static std::pair<CreatorsMapFilterIterator, CreatorsMapFilterIterator>
    makeFilteredRange(CreatorsMap& map, unsigned rank, std::set<TensorDescCreatorTypes> supportedTypes);
    virtual InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision& precision, const InferenceEngine::SizeVector& srcDims) const = 0;
    virtual size_t getMinimalRank() const = 0;
    virtual ~TensorDescCreator() {}
};

class PlainFormatCreator : public TensorDescCreator {
public:
    virtual InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision& precision, const InferenceEngine::SizeVector& srcDims) const;
    virtual size_t getMinimalRank() const { return 0lu; }
};

class PerChannelCreator : public TensorDescCreator {
public:
    virtual InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision &precision, const InferenceEngine::SizeVector &srcDims) const;
    virtual size_t getMinimalRank() const { return 3lu; }
};

class ChannelBlockedCreator : public TensorDescCreator {
public:
    ChannelBlockedCreator(size_t blockSize) : _blockSize(blockSize) {}
    virtual InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision& precision, const InferenceEngine::SizeVector& srcDims) const;
    virtual size_t getMinimalRank() const { return 2lu; }

private:
    size_t _blockSize;
};

class CreatorsMapFilterIterator {
public:
    typedef TensorDescCreator::CreatorsMap::iterator Iterator;
    typedef std::iterator_traits<Iterator>::value_type value_type;
    typedef std::iterator_traits<Iterator>::reference reference;
    typedef std::iterator_traits<Iterator>::pointer pointer;
    typedef std::iterator_traits<Iterator>::difference_type difference_type;
    typedef std::forward_iterator_tag iterator_category;
    typedef std::function<bool(const value_type&)> predicate_type;

public:
    CreatorsMapFilterIterator(predicate_type filter, Iterator begin, Iterator end) : _filter(std::move(filter)), _iter(begin), _end(end)  {}
    CreatorsMapFilterIterator& operator++() {
        do {
            ++_iter;
        } while (_iter != _end && !_filter(*_iter));
        return *this;
    }

    CreatorsMapFilterIterator end() const {
        return CreatorsMapFilterIterator(predicate_type(), _end, _end);
    }

    CreatorsMapFilterIterator operator++(int) {
        CreatorsMapFilterIterator temp(*this);
        ++*this;
        return temp;
    }

    reference operator*() const {
        return *_iter;
    }

    pointer operator->() const {
        return std::addressof(*_iter);
    }

    friend bool operator==(const CreatorsMapFilterIterator& lhs, const CreatorsMapFilterIterator& rhs) {
        return lhs._iter == rhs._iter;
    }

    friend bool operator!=(const CreatorsMapFilterIterator& lhs, const CreatorsMapFilterIterator& rhs) {
        return !(lhs == rhs);
    }

private:
    Iterator _iter;
    Iterator _end;
    predicate_type _filter;
};
} // namespace MKLDNNPlugin