// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_desc_creator.h"
#include <numeric>

using namespace InferenceEngine;
using namespace MKLDNNPlugin;

namespace {
    constexpr size_t channelsPos = 1lu;
}

InferenceEngine::TensorDesc PlainFormatCreator::createDesc(const InferenceEngine::Precision &precision, const InferenceEngine::SizeVector &srcDims) const {
    SizeVector order(srcDims.size());
    std::iota(order.begin(), order.end(), 0);
    return TensorDesc(precision, srcDims, {srcDims, order});
}

InferenceEngine::TensorDesc PerChannelCreator::createDesc(const InferenceEngine::Precision &precision, const InferenceEngine::SizeVector &srcDims) const {
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

    return TensorDesc(precision, srcDims, {blkDims, order});
}

InferenceEngine::TensorDesc ChannelBlockedCreator::createDesc(const InferenceEngine::Precision &precision, const InferenceEngine::SizeVector &srcDims) const {
    if (srcDims.size() < 2) {
        THROW_IE_EXCEPTION << "Can't create blocked tensor descriptor!";
    }

    SizeVector order(srcDims.size());
    std::iota(order.begin(), order.end(), 0);
    order.push_back(channelsPos);

    SizeVector blkDims = srcDims;
    blkDims[channelsPos] = blkDims[channelsPos] / _blockSize + (blkDims[channelsPos] % _blockSize ? 1 : 0);
    blkDims.push_back(_blockSize);

    return TensorDesc(precision, srcDims, {blkDims, order});
}

std::map<TensorDescCreatorTypes, TensorDescCreator::CreatorConstPtr> TensorDescCreator::getCommonCreators() {
    return { { TensorDescCreatorTypes::nspc, CreatorConstPtr(new PerChannelCreator) },
             { TensorDescCreatorTypes::nCsp8c, CreatorConstPtr(new ChannelBlockedCreator(8)) },
             { TensorDescCreatorTypes::nCsp16c, CreatorConstPtr(new ChannelBlockedCreator(16)) },
             { TensorDescCreatorTypes::ncsp, CreatorConstPtr(new PlainFormatCreator) } };
}

std::pair<CreatorsMapFilterIterator, CreatorsMapFilterIterator>
TensorDescCreator::makeFilteredRange(TensorDescCreator::CreatorsMap &map, unsigned int rank) {
    auto rankFilter = [rank](const CreatorsMap::value_type& item) {
        if (item.second->getMinimalRank() > rank) {
            return false;
        }
        return true;
    };

    auto first = CreatorsMapFilterIterator(std::move(rankFilter), map.begin(), map.end());
    auto last = first.end();
    return std::make_pair(first, last);
}

std::pair<CreatorsMapFilterIterator, CreatorsMapFilterIterator>
TensorDescCreator::makeFilteredRange(TensorDescCreator::CreatorsMap &map, unsigned int rank, std::set<TensorDescCreatorTypes> supportedTypes) {
    auto rankTypesFilter = [rank, supportedTypes](const CreatorsMap::value_type& item) {
        if (!supportedTypes.count(item.first)) {
            return false;
        }
        if (item.second->getMinimalRank() > rank) {
            return false;
        }
        return true;
    };

    auto first = CreatorsMapFilterIterator(std::move(rankTypesFilter), map.begin(), map.end());
    auto last = first.end();
    return std::make_pair(first, last);
}
