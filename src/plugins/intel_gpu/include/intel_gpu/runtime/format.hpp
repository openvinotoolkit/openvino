// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <list>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <optional>
#include <vector>
#include <string>
#include <utility>
#include <stdexcept>


namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_memory Memory description and management
/// @{

/// @brief Format information helper class.
struct format_traits {
    /// @brief String representation of a format.
    std::string str;
    /// @brief Number of batch dimensions in a format.
    size_t batch_num;
    /// @brief Number of feature map/channel dimensions in a format.
    size_t feature_num;
    /// @brief Number of spatial (x,y) dimensions in a format.
    size_t spatial_num;
    /// @brief Number of groups in a format.
    size_t group_num;
    /// @brief Dimensions order. Default {0, 1, 2, ... rank }
    std::vector<size_t> _order;
    /// @brief Dimensions changing order from rare to often.
    std::string order;
    /// @brief Dimensions order for internal storage.
    std::string internal_order;
    /// @brief Block sizes as a vector of pairs of dimension number and block size ordered from rare to often.
    std::vector<std::pair<size_t, int>> block_sizes;
    std::vector<std::pair<size_t, int>> logic_block_sizes;
    /// @brief Onednn memory descriptor size used for asymmetric compensation.
    size_t desc_size = 0;
    /// @brief Characters representing batch dimensions in an order.
    static const char* batch_chars() { return "bno"; }
    /// @brief Characters representing feature map/channel dimensions in an order.
    static const char* feature_chars() { return "fic"; }
    /// @brief Characters representing spatial dimensions in an order.
    static const char* spatial_chars() { return "xyzwuvhs"; }
    /// @brief Characters representing group dimensions in an order.
    static const char* group_chars() { return "g"; }
    /// @brief Checks if @p c represents batch dimension.
    static bool is_batch_char(char c) { return std::string(batch_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents feature map/channel dimension.
    static bool is_feature_char(char c) { return std::string(feature_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents spatial dimension.
    static bool is_spatial_char(char c) { return std::string(spatial_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents group dimensions.
    static bool is_group_char(char c) { return std::string(group_chars()).find_first_of(c) != std::string::npos; }

    /// @brief Checks if order has @p c dimension.
    bool has_dimension(char c) const { return order.find_first_of(c) != std::string::npos; }
};

/// @brief Represents memory formats (orders).
/// @n In CNN most of data is described as 4 dimensional blocks. In GPU plugin we describe memory with 4 letters
/// - b - number of blocks in batch. For weights formats: output features - conv, neurons - inner product
/// - f - number of feature maps, features or channels. For weights formats: input features - conv, inputs, inner product
/// - x - spatial, width
/// - y - spatial, height
/// /n
/// For explanation how each format type is implemented in memory we will use naming shown bellow:
struct format {
    enum type : int32_t {
        // Data formats
        bfyx,                                   ///< the most common format for activations in clDNN.
        bfxy,
        bfzyx,                                  ///< format for 5d data tensors
        bfwzyx,                                 ///< batch, feature, 4D spatial
        bfuwzyx,                                ///< 7d tensor
        bfvuwzyx,                               ///< 8d tensor
        yxfb,                                   ///< batch first, feature and than spatials
        byxf,                                   ///< used in bitmaps, input from user i.e b images of RGB format
        fbyx,
        fyxb,                                   ///< format not used inside clDNN, but supported in reorder as extension
        fybx,                                   ///< To be used when onednn gemm allows permute fusing in transformer network. Not for normal use from cldnn.
        xbfy,                                   ///< To be used when onednn gemm allows permute fusing in transformer network. Not for normal use from cldnn.
        ybfx,                                   ///< To be used when onednn gemm allows permute fusing in transformer network. Not for normal use from cldnn.
        bzyxf,
        byfx,                                   ///< To be used when onednn gemm allows permute fusing in transformer network. Not for normal use from cldnn.
        bxfy,                                   ///< To be used when onednn gemm allows permute fusing in transformer network. Not for normal use from cldnn.
                                                ///< for user provided formats.
        b_fs_yx_fsv2,
        b_fs_zyx_fsv2,
        b_fs_yx_fsv4,                           ///< format for input for IMAD convolutions
        b_fs_zyx_fsv4,                          ///< format for input for IMAD 3D convolutions
        b_fs_yx_fsv8,
        b_fs_zyx_fsv8,
        b_fs_yx_fsv16,                          ///< format used for blocked convolution
        b_fs_yx_fsv32,                          ///< format used for blocked int8 convolution
        b_fs_zyx_fsv16,                         ///< format used for 3D blocked convolution (features blocked by 16)
        b_fs_zyx_fsv32,                         ///< format used for blocked int8 3d convolution
        bs_fs_yx_bsv16_fsv32,                   ///< format used for 2D blocked convolution (batch and features blocked by 16 and 32)
        bs_fs_zyx_bsv16_fsv32,                  ///< format used for 3D blocked convolution (batch and features blocked by 16 and 32)
        bs_fs_zyx_bsv16_fsv16,                  ///< format used for 3D blocked convolution (batch and features blocked by 16)
        bs_fs_yx_bsv16_fsv16,                   ///< format used for 2D blocked convolution (batch and features blocked by 16)
        bs_fs_yx_bsv4_fsv4,                     ///< format used for 2D blocked convolution (batch and features blocked by 4)
        bs_fs_yx_bsv8_fsv4,                     ///< format used for 2D blocked convolution (batch and features blocked by 8 and 4)
        bs_fs_zyx_bsv8_fsv4,                    ///< format used for 3D blocked convolution (batch and features blocked by 8 and 4)
        bs_fs_yx_bsv16_fsv4,                    ///< format used for 2D blocked convolution (batch and features blocked by 16 and 4)
        bs_fs_zyx_bsv16_fsv4,                   ///< format used for 3D blocked convolution (batch and features blocked by 16 and 4)
        bs_fs_yx_bsv8_fsv2,                     ///< format used for 2D blocked convolution (batch and features blocked by 8 and 2)
        bs_fs_zyx_bsv8_fsv2,                    ///< format used for 3D blocked convolution (batch and features blocked by 8 and 2)
        bs_fs_yx_bsv16_fsv2,                    ///< format used for 2D blocked convolution (batch and features blocked by 16 and 2)
        bs_fs_zyx_bsv16_fsv2,                   ///< format used for 3D blocked convolution (batch and features blocked by 16 and 2)
        bs_fs_yx_bsv16_fsv8,                    ///< format used for 2D blocked convolution (batch and features blocked by 16 and 8)
        bs_fs_zyx_bsv16_fsv8,                   ///< format used for 3D blocked convolution (batch and features blocked by 16 and 8)
        bs_fs_yx_bsv4_fsv2,                     ///< format used for 2D blocked convolution (batch blocked by 4, features blocked by 2)
        bs_fs_zyx_bsv4_fsv4,                    ///< format used for 3D blocked convolution (batch and features blocked by 4)
        bs_fs_zyx_bsv4_fsv2,                    ///< format used for 3D blocked convolution (batch blocked by 4, features blocked by 2)
        bs_fs_yx_bsv32_fsv32,                   ///< format used for big batches (batch and features blocked by 32)
        bs_fs_yx_bsv32_fsv16,                   ///< format used for big batches (batch blocked by 32, features blocked by 16)
        bs_fs_zyx_bsv32_fsv32,                  ///< format used for big batches (batch and features blocked by 32)
        bs_fs_zyx_bsv32_fsv16,                  ///< format used for big batches (batch blocked by 32, features blocked by 16)
        fs_b_yx_fsv32,                          ///< format for input for fp16 primitives
        bs_fs_fsv8_bsv8,                        ///< format used only for fully connected
        bs_fs_fsv8_bsv16,                       ///< format used only for fully connected
        bs_f_bsv16,                             ///< format used only for fully connected weights fp16 batch=1 : bs - batch slice
                                                ///< (responses slice), bsv16 - 16 values of single batch slice, f - flattened plane of (fyx)
        winograd_2x3_s1_data,                   ///< format used for input for winograd convolution, F(2,3) -- filter 3x3 with stride 1
        nv12,                                   ///< format for media nv12 input
        image_2d_rgba,                          ///< format for image2d RGBA, always allocates memory for 4 feature maps (even when only 3 are used)

        // Weights formats
        oiyx,                                         ///< the most common format for 2D weights
        ioyx,                                         ///< 2D weights format for deconvolutions
        yxio,                                         ///< format used 2D weights
        oizyx,                                        ///< the most common format for 3D convolution
        iozyx,                                        ///< 3D weights format for deconvolutions
        iyxo,
        oyxi,
        oyix,
        oxiy,
        os_iyx_osv16,                                 ///< format used only for convolution weights
        o_is_yx_isv4,                                 ///< format used only for convolution weights
        o_is_yx_isv16,                                ///< format used only for convolution weights
        os_is_yx_osv16_isv16,                         ///< format used for convolution i8 weights
        os_is_zyx_osv32_isv16,
        os_is_yx_osv32_isv2,                          ///< format used only for fully connected weights compressed for i4
        os_is_zyx_osv64_isv16,
        os_is_yx_osv64_isv2,                          ///< format used only for fully connected weights compressed for i4
        os_zyxi_osv16,                                ///< format used for weights for 3D convolution
        os_is_yx_isv16_osv16,                         ///< format used for blocked convolution
        os_is_zyx_isv16_osv16,                        ///< format used for weights for blocked 3D convolution
        is_os_zyx_isv16_osv16,                        ///< format used for weights for blocked 3D deconvolution
        is_os_yx_isv16_osv16,                         ///< format used for weights for blocked deconvolution
        os_is_yx_isv8_osv16_isv2,                     ///< format used for weights for blocked 2D convolution
        os_is_zyx_isv8_osv16_isv2,                    ///< format used for weights for blocked 3D convolution
                                                      ///< os - output feature maps slice, i - input feature maps,
                                                      ///< yx - spatials, sv16 - 16 values of single slice.
        os_iyx_osv32,                                 ///< format used only for convolution weights:
                                                      ///< os - output feature maps slice, i - input feature maps,
                                                      ///< yx - spatials, sv32 - 32 values of single slice.
        os_iyx_osv64,                                 ///< format used only for convolution weights:
                                                      ///< os - output feature maps slice, i - input feature maps,
                                                      ///< yx - spatials, sv64 - 64 values of single slice.
        image_2d_weights_c4_fyx_b,                    ///< image format for weights, width size is f*y*x/4
                                                      ///< (4-channels filled with fyx data), height is b
        image_2d_weights_c1_b_fyx,                    ///< image format for weights, width size is b,
                                                      ///< height is f*y*x, single channel
        winograd_2x3_s1_weights,                      ///< format used for weights for winograd non-fused
                                                      ///< convolution, F(2,3) -- filter 3x3 with stride 1
        winograd_2x3_s1_fused_weights,                ///< format used for weights for winograd fused
                                                      ///< convolution, F(2,3) -- filter 3x3 with stride 1
        winograd_6x3_s1_fused_weights,                ///< format used for weights for winograd fused
                                                      ///< convolution, F(6,3) -- filter 3x3 with stride 1
        image_2d_weights_winograd_6x3_s1_fbxyb,       ///< image format used for weights for winograd fused
                                                      ///< convolution, F(6,3) -- filter 3x3 with stride 1
        image_2d_weights_winograd_6x3_s1_xfbyb,       ///< image format used for weights for winograd fused
                                                      ///< convolution, F(6,3) -- filter 3x3 with stride 1
        os_is_yx_isa8_osv8_isv4,                      ///< format for weights for MMAD convolution
        os_is_zyx_isa8_osv8_isv4,                     ///< format for weights for MMAD convolution
        os_is_yx_isa8_osv16_isv4,                     ///< format for weights for fully connected MMAD
        os_is_zyx_isa8_osv16_isv4,                    ///< format for weights for fully connected MMAD
        os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,   ///< format for weights for MMAD fsv32 convolution
        os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4,  ///< format for weights for MMAD fsv32 convolution
        os_is_zyx_osa4_isa8_osv8_isv4,                ///< format for weights for MMAD fsv32 convolution
        os_is_yx_osa4_isa8_osv8_isv4,                 ///< format for weights for MMAD fsv32 convolution
        os_is_yx_osv16_isv4,                          ///< format for weights for IMAD convolutions
        os_is_yx_osv8_isv4,                           ///< format used for convolution i8 weights
        os_is_zyx_osv16_isv16,                        ///< format for weights for IMAD convolutions
        os_is_yx_osv32_isv4_swizzled_by_2,            ///< format for weights for IMAD convolutions
        os_is_yx_osv32_isv4,                          ///< format for weights for IMAD convolutions
        os_is_zyx_osv32_isv4,                         ///< format for weights for IMAD convolutions
        os_iyx_osv8,
        os_iyx_osv32__ai32,
        iy_xs_os_xsv2_osv8__ao32,
        iy_xs_os_xsv2_osv16__ao32,
        i_yxs_os_yxsv2_osv16,
        os_i_osv16,                                   ///< format used only for fully connected weights
        os_i_osv16__ai8,                              ///< format used only for fully connected weights
        os_i_osv8__ai8,                               ///< format used only for fully connected weights

        goiyx,                                        ///< format used for weights for 2D convolution
        gioyx,                                        ///< format used for weights for 2D deconvolution
        gyxio,                                        ///< format used for weights for 2D convolution
        goizyx,                                       ///< format used for weights for 3D convolution
        giozyx,                                       ///< format used for weights for 3D deconvolution
        g_os_iyx_osv8,                                ///< format used for weights for 2D convolution
        g_os_iyx_osv16,                               ///< format used for weights for 2D convolution
        g_os_iyx_osv32,                               ///< format used for weights for 2D convolution
        gs_oiyx_gsv16,                                ///< format used for weights for 2D convolution
        gs_oizyx_gsv16,                               ///< format used for weights for 3D convolution
        gs_oiyx_gsv32,                                ///< format used for weights for 2D convolution
        g_is_os_zyx_isv16_osv16,                      ///< format used for grouped weights for blocked 3D deconvolution
        g_os_is_yx_osv16_isv4,
        g_os_is_zyx_osv16_isv16,
        g_is_os_yx_isv16_osv16,
        g_os_is_zyx_isv8_osv16_isv2,
        g_os_is_yx_isv8_osv16_isv2,
        g_os_is_zyx_isv16_osv16,
        g_os_zyx_is_osv16_isv4,                       ///< format for imad deconvolution
        g_os_zyx_is_osv16_isv16,                      ///< format for imad deconvolution
        g_os_zyx_is_osv16_isv32,                      ///< format for imad deconvolution
        g_os_zyx_is_osv32_isv4,                       ///< format for imad deconvolution
        g_os_zyx_is_osv32_isv16,                      ///< format for imad deconvolution
        g_os_zyx_is_osv32_isv32,                      ///< format for imad deconvolution
        g_os_is_yx_isv16_osv16,
        gs_oi_yxs_gsv4_yxsv4,
        gs_oi_yxs_gsv16_yxsv4,
        gs_oi_yxs_gsv32_yxsv4,
        gi_yxs_os_yxsv2_osv16,
        giy_xs_os_xsv2_osv8__ao32,
        giy_xs_os_xsv2_osv16__ao32,

        format_num,  ///< number of format types
        custom,      ///< means that this format is created based on custom format traits and may have no corresponding label
        any        = -1
    };

    /// @brief Get format traits for particular @p format::type
    static const format_traits& traits(type fmt);

    /// @brief Get traits for current format
    const format_traits& traits() const;
    /// @brief Returns number of batch dimensions for a @p format.
    static size_t batch_num(const format& fmt) { return fmt.traits().batch_num; }
    /// @brief Returns number of feature dimensions for a @p format.
    static size_t feature_num(const format& fmt) { return fmt.traits().feature_num; }
    /// @brief Returns number of spatial dimensions for a @p format.
    static size_t spatial_num(const format& fmt) { return fmt.traits().spatial_num; }
    /// @brief Returns number of group dimensions for a @p format.
    static size_t group_num(const format& fmt) { return fmt.traits().group_num; }
    /// @brief Returns an order of dimensions for a @ format.
    static const std::string& order(const format& fmt) { return fmt.traits().order; }
    /// @brief Returns an internal orders of dimensions for a @p format.
    static const std::string& internal_order(const format& fmt) { return fmt.traits().internal_order; }
    /// @brief Returns block sizes for @p format.
    static const std::vector<std::pair<size_t, int>>& block_sizes(const format& fmt) { return fmt.traits().block_sizes; }
    static const std::vector<std::pair<size_t, int>>& logic_block_sizes(const format& fmt) { return fmt.traits().logic_block_sizes; }
    /// @brief Returns number of dimensions contained within a @p format
    static size_t dimension(const format& fmt) { return order(fmt).size(); }
    /// @brief Checks if @p format is a winograd format
    static bool is_winograd(const format& fmt) {
        return (fmt == winograd_2x3_s1_data ||
                fmt == winograd_2x3_s1_weights ||
                fmt == winograd_2x3_s1_fused_weights ||
                fmt == winograd_6x3_s1_fused_weights ||
                fmt == image_2d_weights_winograd_6x3_s1_fbxyb ||
                fmt == image_2d_weights_winograd_6x3_s1_xfbyb); }
    /// @brief Checks if @p format is of image2d type
    static bool is_image_2d(const format& fmt) {
        return (fmt == image_2d_weights_c4_fyx_b ||
                fmt == image_2d_weights_c1_b_fyx ||
                fmt == image_2d_weights_winograd_6x3_s1_fbxyb ||
                fmt == image_2d_weights_winograd_6x3_s1_xfbyb ||
                fmt == nv12 ||
                fmt == image_2d_rgba);
    }
    /// @brief Checks if @p format is weights format
    static bool is_weights_format(const format& fmt) {
        if (fmt == format::custom)
            return true;
        const auto internal_order = fmt.traits().internal_order;
        const auto weights_chars = { "o", "i" };
        for (const auto& c : weights_chars) {
            if (internal_order.find_first_of(c) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
    /// @brief Checks if @p format is simple data format
    static bool is_simple_data_format(const format& fmt) {
        return (fmt == yxfb || fmt == byxf || fmt == byfx || fmt == bxfy || fmt == bfyx || fmt == fyxb || fmt == fybx ||
                fmt == bfxy ||fmt == xbfy || fmt == ybfx || fmt == fbyx || fmt == bfzyx || fmt == bfwzyx || fmt == bfuwzyx ||
                fmt == bfvuwzyx);
    }

    static format get_default_format(size_t rank, bool is_weights = false, bool is_grouped = false);
    static bool is_default_format(const format& fmt);

    static format adjust_to_rank(format fmt, size_t new_rank);

    static const std::vector<std::pair<size_t, int>> per_axis_block_size(format fmt);

    static format find_format(const std::vector<uint64_t>& order,
                              const std::vector<std::pair<size_t, int>>& block_sizes,
                              bool is_weights = false,
                              bool is_grouped = false,
                              bool is_image_2d = false,
                              bool is_winograd = false,
                              bool is_nv12 = false);

    /// @brief Checks if @p format is of grouped type
    static bool is_grouped(const format& fmt) { return group_num(fmt) != 0; }
    /// @brief Checks if @p format is of image type
    static bool is_image(const format& fmt) { return (is_image_2d(fmt)); }
    /// @brief Checks if @p format is blocked format
    static bool is_blocked(const format& fmt) { return !(block_sizes(fmt).empty()); }
    /// @brief Checks if @p format is blocked format which has single inner block
    static bool is_single_blocked(const format& fmt) { return (block_sizes(fmt).size() == 1); }
    /// @brief Checks if @p format is blocked format which has multiple inner blocks
    static bool is_multi_blocked(const format& fmt) { return (block_sizes(fmt).size() > 1); }
    /// @brief Checks if @p format is nv12 format
    static bool is_nv12(const format& fmt) { return (fmt == nv12); }

    /// @brief Returns number of batch dimensions.
    size_t batch_num() const { return traits().batch_num; }
    /// @brief Returns number of feature dimensions.
    size_t feature_num() const { return traits().feature_num; }
    /// @brief Returns number of spatial dimensions.
    size_t spatial_num() const { return traits().spatial_num; }
    /// @brief Returns number of group dimensions.
    size_t group_num() const { return traits().group_num; }
    /// @brief Returns an order of dimensions.
    const std::vector<uint64_t>& dims_order() const { return traits()._order; }
    /// @brief Returns an order of dimensions in form of string.
    const std::string& order() const { return traits().order; }
    /// @brief Returns an internal orders of dimensions form of string.
    const std::string& internal_order() const { return traits().internal_order; }
    /// @brief Returns block sizes as vector of pairs of dimension and block size for that dimension.
    const std::vector<std::pair<size_t, int>>& block_sizes() const { return traits().block_sizes; }
    const std::vector<std::pair<size_t, int>>& logic_block_sizes() const { return traits().logic_block_sizes; }
    /// @brief Returns number of dimensions contained within this format
    size_t dimension() const { return traits()._order.size(); }
    /// @brief Checks if @p format is a winograd format
    bool is_winograd() const { return is_winograd(*this); }
    /// @brief Checks if @p format is of image 2d type
    bool is_image_2d() const { return is_image_2d(*this); }
    /// @brief Checks if @p format is of image type
    bool is_image() const { return is_image(*this); }
    /// @brief Checks if @p format is blocked format
    bool is_blocked() { return is_blocked(*this); }
    /// @brief Checks if @p format is a nv12 format
    bool is_nv12() const { return is_nv12(*this); }

    /// @brief Transforms dimension from internal order to external order
    size_t internal_to_external(size_t idx) const {
        auto index = order().find_first_of(internal_order()[idx]);
        if (index == std::string::npos)
            throw std::invalid_argument("Internal dimension index does not map to external index.");
        return index;
    }

    type value;

    std::optional<format_traits> custom_traits = {};

    /// @brief Implicit conversion from format::type.
    format(type t) : value(t) {}

    /// @brief custom format from format_traits.
    explicit format(const format_traits& traits) : value(format::custom), custom_traits(traits) {}

    /// @brief Implicit conversion to format::type.
    constexpr operator type() const { return value; }

    std::string to_string() const;
};

inline std::ostream& operator<<(std::ostream& os, const format& fmt) {
    return os << fmt.to_string();
}

/// @}
/// @}
}  // namespace cldnn
