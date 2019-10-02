// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This header file provides structures to store info about pre-processing of network inputs (scale, mean image, ...)
 * @file ie_preprocess.hpp
 */
#pragma once

#include "ie_blob.h"
#include <vector>
#include <memory>

namespace InferenceEngine {

/**
 * @brief This structure stores info about pre-processing of network inputs (scale, mean image, ...)
 */
struct PreProcessChannel {
    /** @brief Scale parameter for a channel */
    float stdScale = 1;

    /** @brief Mean value for a channel */
    float meanValue = 0;

    /** @brief Mean data for a channel */
    Blob::Ptr meanData;

    /** @brief Smart pointer to an instance */
    using Ptr = std::shared_ptr<PreProcessChannel>;
};

/**
 * @brief Defines available types of mean
 */
enum MeanVariant {
    MEAN_IMAGE, /**< mean value is specified for each input pixel */
    MEAN_VALUE, /**< mean value is specified for each input channel */
    NONE,       /**< no mean value specified */
};

/**
 * @enum ResizeAlgorithm
 * @brief Represents the list of supported resize algorithms.
 */
enum ResizeAlgorithm {
    NO_RESIZE = 0,
    RESIZE_BILINEAR,
    RESIZE_AREA
};

/**
 * @brief This class stores pre-process information for the input
 */
class PreProcessInfo {
    // Channel data
    std::vector<PreProcessChannel::Ptr> _channelsInfo;
    MeanVariant _variant = NONE;

    // Resize Algorithm to be applied for input before inference if needed.
    ResizeAlgorithm _resizeAlg = NO_RESIZE;

    // Color format to be used in on-demand color conversions applied to input before inference
    ColorFormat _colorFormat = ColorFormat::RAW;

public:
    /**
     * @brief Overloaded [] operator to safely get the channel by an index
     * Throws an exception if channels are empty
     * @param index Index of the channel to get
     * @return The pre-process channel instance
     */
    PreProcessChannel::Ptr &operator[](size_t index) {
        if (_channelsInfo.empty()) {
            THROW_IE_EXCEPTION << "accessing pre-process when nothing was set.";
        }
        if (index >= _channelsInfo.size()) {
            THROW_IE_EXCEPTION << "pre process index " << index << " is out of bounds.";
        }
        return _channelsInfo[index];
    }

    /**
     * @brief operator [] to safely get the channel preprocessing information by index.
     * Throws exception if channels are empty or index is out of border
     *
     * @param index Index of the channel to get
     * @return The const preprocess channel instance
     */
    const PreProcessChannel::Ptr &operator[](size_t index) const {
        if (_channelsInfo.empty()) {
            THROW_IE_EXCEPTION << "accessing pre-process when nothing was set.";
        }
        if (index >= _channelsInfo.size()) {
            THROW_IE_EXCEPTION << "pre process index " << index << " is out of bounds.";
        }
        return _channelsInfo[index];
    }

    /**
     * @brief Returns a number of channels to preprocess
     * @return The number of channels
     */
    size_t getNumberOfChannels() const {
        return _channelsInfo.size();
    }

    /**
     * @brief Initializes with given number of channels
     * @param numberOfChannels Number of channels to initialize
     */
    void init(const size_t numberOfChannels) {
        _channelsInfo.resize(numberOfChannels);
        for (auto &channelInfo : _channelsInfo) {
            channelInfo = std::make_shared<PreProcessChannel>();
        }
    }

    /**
     * @brief Sets mean image values if operation is applicable.
     * Also sets the mean type to MEAN_IMAGE for all channels
     * @param meanImage Blob with a mean image
     */
    void setMeanImage(const Blob::Ptr &meanImage) {
        if (meanImage.get() == nullptr) {
            THROW_IE_EXCEPTION << "Failed to set invalid mean image: nullptr";
        } else if (meanImage.get()->getTensorDesc().getLayout() != Layout::CHW) {
            THROW_IE_EXCEPTION << "Mean image layout should be CHW";
        } else if (meanImage.get()->getTensorDesc().getDims().size() != 3) {
            THROW_IE_EXCEPTION << "Failed to set invalid mean image: number of dimensions != 3";
        } else if (meanImage.get()->getTensorDesc().getDims()[0] != getNumberOfChannels()) {
            THROW_IE_EXCEPTION << "Failed to set invalid mean image: number of channels != "
                               << getNumberOfChannels();
        }
        _variant = MEAN_IMAGE;
    }

    /**
     * @brief Sets mean image values if operation is applicable.
     * Also sets the mean type to MEAN_IMAGE for a particular channel
     * @param meanImage Blob with a mean image
     * @param channel Index of a particular channel
     */
    void setMeanImageForChannel(const Blob::Ptr &meanImage, const size_t channel) {
        if (meanImage.get() == nullptr) {
            THROW_IE_EXCEPTION << "Failed to set invalid mean image for channel: nullptr";
        } else if (meanImage.get()->getTensorDesc().getDims().size() != 2) {
            THROW_IE_EXCEPTION << "Failed to set invalid mean image for channel: number of dimensions != 2";
        } else if (channel >= _channelsInfo.size()) {
            THROW_IE_EXCEPTION << "Channel " << channel << " exceed number of PreProcess channels: "
                               << _channelsInfo.size();
        }
        _variant = MEAN_IMAGE;
        _channelsInfo[channel]->meanData = meanImage;
    }

    /**
     * @brief Sets a type of mean operation
     * @param variant Type of mean operation to set
     */
    void setVariant(const MeanVariant &variant) {
        _variant = variant;
    }

    /**
     * @brief Gets a type of mean operation
     * @return The type of mean operation
     */
    MeanVariant getMeanVariant() const {
        return _variant;
    }

    /**
     * @brief Sets resize algorithm to be used during pre-processing.
     * @param alg Resize algorithm.
     */
    void setResizeAlgorithm(const ResizeAlgorithm &alg) {
        _resizeAlg = alg;
    }

    /**
     * @brief Gets preconfigured resize algorithm.
     * @return Resize algorithm.
     */
    ResizeAlgorithm getResizeAlgorithm() const {
        return _resizeAlg;
    }

    /**
     * @brief Changes the color format of the input data provided by the user
     * This function should be called before loading the network to the plugin
     * Setting color format different from ColorFormat::RAW enables automatic color conversion
     * (as a part of built-in preprocessing routine)
     * @param fmt A new color format associated with the input
     */
    void setColorFormat(ColorFormat fmt) {
        _colorFormat = fmt;
    }

    /**
     * @brief Gets a color format associated with the input
     * @details By default, the color format is ColorFormat::RAW meaning
     *          there is no particular color format assigned to the input
     * @return Color format.
     */
    ColorFormat getColorFormat() const {
        return _colorFormat;
    }
};
}  // namespace InferenceEngine
