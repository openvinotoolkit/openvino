// Copyright (C) 2018-2025 Intel Corporation
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
    case ColorFormat::I420_THREE_PLANES:
        name = "I420 (multi-plane)";
        break;
    case ColorFormat::I420_SINGLE_PLANE:
        name = "I420 (single plane)";
        break;
    case ColorFormat::RGBX:
        name = "RGBX";
        break;
    case ColorFormat::BGRX:
        name = "BGRX";
        break;
    case ColorFormat::GRAY:
        name = "GRAY";
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

    PartialShape shape(size_t plane_num, const PartialShape& src_shape, const Layout& src_layout) const {
        OPENVINO_ASSERT(plane_num < planes_count(),
                        "Internal error: incorrect plane number specified for color format");
        return calculate_shape(plane_num, src_shape, src_layout);
    }

protected:
    virtual PartialShape calculate_shape(size_t plane_num,
                                         const PartialShape& src_shape,
                                         const Layout& src_layout) const {
        return src_shape;
    }

    explicit ColorFormatInfo(ColorFormat format) : m_format(format) {}
    ColorFormat m_format;
};

// --- Derived classes ---
class ColorFormatNHWC : public ColorFormatInfo {
public:
    explicit ColorFormatNHWC(ColorFormat format) : ColorFormatInfo(format) {}

    Layout default_layout() const override {
        return "NHWC";
    }
};

// Assume that it should work with NHWC and NCHW cases, but default is NHWC
class ColorFormatRGB : public ColorFormatNHWC {
public:
    explicit ColorFormatRGB(ColorFormat format) : ColorFormatNHWC(format) {}

protected:
    PartialShape calculate_shape(size_t plane_num,
                                 const PartialShape& src_shape,
                                 const Layout& src_layout) const override {
        PartialShape result = src_shape;
        if (src_shape.rank().is_static()) {
            auto c_idx = ov::layout::channels_idx(src_layout);
            c_idx = c_idx < 0 ? c_idx + src_shape.size() : c_idx;
            result[c_idx] = 3;
        }
        return result;
    }
};

using ColorFormatBGR = ColorFormatRGB;

// Applicable for both NV12 and I420 formats
class ColorFormatInfoYUV420_Single : public ColorFormatNHWC {
public:
    explicit ColorFormatInfoYUV420_Single(ColorFormat format) : ColorFormatNHWC(format) {}

protected:
    PartialShape calculate_shape(size_t plane_num,
                                 const PartialShape& src_shape,
                                 const Layout& src_layout) const override {
        PartialShape result = src_shape;
        if (src_shape.rank().is_static() && src_shape.rank().get_length() == 4) {
            result[3] = 1;
            if (result[1].is_static()) {
                result[1] = result[1].get_length() * 3 / 2;
            }
        }
        return result;
    }
};

class ColorFormatInfoNV12_TwoPlanes : public ColorFormatNHWC {
public:
    explicit ColorFormatInfoNV12_TwoPlanes(ColorFormat format) : ColorFormatNHWC(format) {}

    size_t planes_count() const override {
        return 2;
    }

protected:
    PartialShape calculate_shape(size_t plane_num,
                                 const PartialShape& src_shape,
                                 const Layout& src_layout) const override {
        PartialShape result = src_shape;
        if (src_shape.rank().is_static() && src_shape.rank().get_length() == 4) {
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
};

class ColorFormatInfoI420_ThreePlanes : public ColorFormatNHWC {
public:
    explicit ColorFormatInfoI420_ThreePlanes(ColorFormat format) : ColorFormatNHWC(format) {}

    size_t planes_count() const override {
        return 3;
    }

protected:
    PartialShape calculate_shape(size_t plane_num,
                                 const PartialShape& src_shape,
                                 const Layout& src_layout) const override {
        PartialShape result = src_shape;
        if (src_shape.rank().is_static() && src_shape.rank().get_length() == 4) {
            result[3] = 1;  //  Number of channels is always 1 for I420 planes
            if (plane_num == 0) {
                return result;
            } else {
                // UV plane has half or width and half of height.
                if (result[1].is_static()) {
                    result[1] = result[1].get_length() / 2;
                }
                if (result[2].is_static()) {
                    result[2] = result[2].get_length() / 2;
                }
            }
        }
        return result;
    }
};

class ColorFormatInfo_RGBX_Base : public ColorFormatNHWC {
public:
    explicit ColorFormatInfo_RGBX_Base(ColorFormat format) : ColorFormatNHWC(format) {}

protected:
    PartialShape calculate_shape(size_t plane_num,
                                 const PartialShape& src_shape,
                                 const Layout& src_layout) const override {
        PartialShape result = src_shape;
        if (src_shape.rank().is_static() && src_shape.rank().get_length() == 4) {
            result[3] = 4;
            return result;
        }
        return result;
    }
};

}  // namespace preprocess
}  // namespace ov
