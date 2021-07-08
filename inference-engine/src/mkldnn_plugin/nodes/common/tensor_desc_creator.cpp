// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_desc_creator.h"
#include <numeric>

using namespace InferenceEngine;
using namespace MKLDNNPlugin;

namespace {
constexpr size_t channelsPos = 1lu;

class PlainFormatCreator : public TensorDescCreator {
public:
    InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision& precision, const InferenceEngine::SizeVector& srcDims) const override {
        SizeVector order(srcDims.size());
        std::iota(order.begin(), order.end(), 0);
//        return BlockedMemoryDesc(precision, srcDims, {srcDims, order});
        return BlockedMemoryDesc(precision, srcDims, srcDims, order);
    }
    size_t getMinimalRank() const override { return 0lu; }
};

class PerChannelCreator : public TensorDescCreator {
public:
<<<<<<< HEAD
    InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision &precision, const InferenceEngine::SizeVector &srcDims) const override {
=======
    virtual BlockedMemoryDesc createDesc(const InferenceEngine::Precision &precision, const InferenceEngine::SizeVector &srcDims) const {
>>>>>>> [CPU] Unified memory descriptor: initial commit
        SizeVector order(srcDims.size());
        std::iota(order.begin(), order.end(), 0);
        SizeVector blkDims = srcDims;
        if (srcDims.size() > 2) {
            auto moveElementBack = [](SizeVector& vector, size_t indx) {
                auto itr = vector.begin() + indx;
                std::rotate(itr, itr + 1, vector.end());
            };

            moveElementBack(order, channelsPos);
            moveElementBack(blkDims, channelsPos);
        }

//        return BlockedMemoryDesc(precision, srcDims, {blkDims, order});
        return BlockedMemoryDesc(precision, srcDims, blkDims, order);
    }
    size_t getMinimalRank() const override { return 3lu; }
};

class ChannelBlockedCreator : public TensorDescCreator {
public:
    ChannelBlockedCreator(size_t blockSize) : _blockSize(blockSize) {}
<<<<<<< HEAD
    InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision& precision, const InferenceEngine::SizeVector& srcDims) const override {
=======
    virtual BlockedMemoryDesc createDesc(const InferenceEngine::Precision& precision, const InferenceEngine::SizeVector& srcDims) const {
>>>>>>> [CPU] Unified memory descriptor: initial commit
        if (srcDims.size() < 2) {
            IE_THROW() << "Can't create blocked tensor descriptor!";
        }

        SizeVector order(srcDims.size());
        std::iota(order.begin(), order.end(), 0);
        order.push_back(channelsPos);

        SizeVector blkDims = srcDims;
        blkDims[channelsPos] = blkDims[channelsPos] / _blockSize + (blkDims[channelsPos] % _blockSize ? 1 : 0);
        blkDims.push_back(_blockSize);

//        return BlockedMemoryDesc(precision, srcDims, {blkDims, order});
        return BlockedMemoryDesc(precision, srcDims, blkDims, order);
    }
    size_t getMinimalRank() const override { return 3lu; }

private:
    size_t _blockSize;
};
} // namespace

const TensorDescCreator::CreatorsMap& TensorDescCreator::getCommonCreators() {
    static const CreatorsMap map{ { TensorDescCreatorTypes::nspc, CreatorConstPtr(new PerChannelCreator) },
                                { TensorDescCreatorTypes::nCsp8c, CreatorConstPtr(new ChannelBlockedCreator(8)) },
                                { TensorDescCreatorTypes::nCsp16c, CreatorConstPtr(new ChannelBlockedCreator(16)) },
                                { TensorDescCreatorTypes::ncsp, CreatorConstPtr(new PlainFormatCreator) } };
    return map;
}

std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator>
TensorDescCreator::makeFilteredRange(const CreatorsMap &map, unsigned int rank) {
    auto rankFilter = [rank](const CreatorsMap::value_type& item) {
        if (item.second->getMinimalRank() > rank) {
            return false;
        }
        return true;
    };

    auto first = CreatorsMapFilterConstIterator(std::move(rankFilter), map.begin(), map.end());
    auto last = first.end();
    return std::make_pair(first, last);
}

std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator>
TensorDescCreator::makeFilteredRange(const CreatorsMap& map, unsigned rank, const std::vector<TensorDescCreatorTypes>& supportedTypes) {
    unsigned bitMask = 0ul;
    for (auto& item : supportedTypes) {
        bitMask |= 1 << static_cast<unsigned>(item);
    }

    auto rankTypesFilter = [rank, bitMask](const CreatorsMap::value_type& item) {
        if (!(bitMask & (1 << static_cast<unsigned>(item.first)))) {
            return false;
        }
        if (item.second->getMinimalRank() > rank) {
            return false;
        }
        return true;
    };

    auto first = CreatorsMapFilterConstIterator(std::move(rankTypesFilter), map.begin(), map.end());
    auto last = first.end();
    return std::make_pair(first, last);
}

std::pair<CreatorsMapFilterConstIterator, CreatorsMapFilterConstIterator>
TensorDescCreator::makeFilteredRange(const CreatorsMap &map, TensorDescCreator::Predicate predicate) {
    auto first = CreatorsMapFilterConstIterator(std::move(predicate), map.begin(), map.end());
    auto last = first.end();
    return std::make_pair(first, last);
}
