// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layouts.h>

namespace MKLDNNPlugin {

enum class TensorDescCreatorTypes {
    plain,
    perChannel,
    channelBlocked8,
    channelBlocked16
};

class TensorDescCreator {
public:
    typedef std::shared_ptr<TensorDescCreator> CreatorPtr;
    typedef std::shared_ptr<const TensorDescCreator> CreatorConstPtr;

public:
    static std::map<TensorDescCreatorTypes, CreatorConstPtr> getCommonCreators();
    virtual InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision& precision, const InferenceEngine::SizeVector& srcDims) const = 0;
    virtual ~TensorDescCreator() {}
};

class PlainFormatCreator : public TensorDescCreator {
public:
    virtual InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision& precision, const InferenceEngine::SizeVector& srcDims) const;
};

class PerChannelCreator : public TensorDescCreator {
public:
    virtual InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision& precision, const InferenceEngine::SizeVector& srcDims) const;
};

class ChannelBlockedCreator : public TensorDescCreator {
public:
    ChannelBlockedCreator(size_t blockSize) : _blockSize(blockSize) {}
    virtual InferenceEngine::TensorDesc createDesc(const InferenceEngine::Precision& precision, const InferenceEngine::SizeVector& srcDims) const;

private:
    size_t _blockSize;
};
} // namespace MKLDNNPlugin