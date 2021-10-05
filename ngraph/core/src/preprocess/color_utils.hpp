// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/layout.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/preprocess/color_format.hpp"

namespace ov {
namespace preprocess {

/// \brief Helper function to check if color format represents RGB family
inline bool is_rgb_family(const ColorFormat& format) {
    return format == ColorFormat::RGB || format == ColorFormat::BGR;
}

inline std::string color_format_name(ColorFormat format) {
    std::string name;
    switch (format) {
    case ColorFormat::RGB:
        name = "RGB";
        break;
    case ColorFormat::BGR:
        name = "BGR";
        break;
    case ColorFormat::NV12_TWO_PLANES:
        name = "NV12 (multi-plane)";
        break;
    case ColorFormat::NV12_SINGLE_PLANE:
        name = "NV12 (single plane)";
        break;
    default:
        name = "Unknown";
        break;
    }
    return name;
}

/// \brief Internal helper class to get information depending on color format
class ColorFormatInfo {
public:
    static std::unique_ptr<ColorFormatInfo> get(ColorFormat format);

    virtual ~ColorFormatInfo() = default;

    virtual size_t planes_count() const {
        return 1;
    }

    virtual Layout default_layout() const {
        return {};
    }

    // Calculate shape of plane based image shape in NHWC format
    PartialShape shape(size_t plane_num, const PartialShape& image_src_shape) const {
        OPENVINO_ASSERT(plane_num < planes_count(),
                        "Internal error: incorrect plane number specified for color format");
        return calculate_shape(plane_num, image_src_shape);
    }

    std::string friendly_suffix(size_t plane_num) const {
        OPENVINO_ASSERT(plane_num < planes_count(),
                        "Internal error: incorrect plane number specified for color format");
        return calc_name_suffix(plane_num);
    }

protected:
    virtual PartialShape calculate_shape(size_t plane_num, const PartialShape& image_shape) const {
        return image_shape;
    }
    virtual std::string calc_name_suffix(size_t plane_num) const {
        return {};
    }
    explicit ColorFormatInfo(ColorFormat format) : m_format(format) {}
    ColorFormat m_format;
};

// --- Derived classes ---
class ColorFormatInfoNV12_Single : public ColorFormatInfo {
public:
    explicit ColorFormatInfoNV12_Single(ColorFormat format) : ColorFormatInfo(format) {}

protected:
    PartialShape calculate_shape(size_t plane_num, const PartialShape& image_shape) const override {
        PartialShape result = image_shape;
        if (image_shape.rank().is_static() && image_shape.rank().get_length() == 4) {
            result[3] = 1;
            if (result[1].is_static()) {
                result[1] = result[1].get_length() * 3 / 2;
            }
        }
        return result;
    }

    Layout default_layout() const override {
        return "NHWC";
    }
};

class ColorFormatInfoNV12_TwoPlanes : public ColorFormatInfo {
public:
    explicit ColorFormatInfoNV12_TwoPlanes(ColorFormat format) : ColorFormatInfo(format) {}

    size_t planes_count() const override {
        return 2;
    }

protected:
    PartialShape calculate_shape(size_t plane_num, const PartialShape& image_shape) const override {
        PartialShape result = image_shape;
        if (image_shape.rank().is_static() && image_shape.rank().get_length() == 4) {
            if (plane_num == 0) {
                result[3] = 1;
                return result;
            } else {
                // UV plane has half or width and half of height. Number of channels is 2
                if (result[1].is_static()) {
                    result[1] = result[1].get_length() / 2;
                }
                if (result[2].is_static()) {
                    result[2] = result[2].get_length() / 2;
                }
                result[3] = 2;
            }
        }
        return result;
    }
    std::string calc_name_suffix(size_t plane_num) const override {
        if (plane_num == 0) {
            return "/Y";
        }
        return "/UV";
    }

    Layout default_layout() const override {
        return "NHWC";
    }
};

}  // namespace preprocess
}  // namespace ov
