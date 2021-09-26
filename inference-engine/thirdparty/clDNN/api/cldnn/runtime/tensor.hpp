// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compounds.hpp"
#include "utils.hpp"

#include <map>
#include <list>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include <functional>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_memory Memory description and management
/// @{

/// @brief Format information helper class.
struct format_traits {
    /// @brief Number of batch dimensions in a format.
    size_t batch_num;
    /// @brief Number of feature map/channel dimensions in a format.
    size_t feature_num;
    /// @brief Number of spatial (x,y) dimensions in a format.
    size_t spatial_num;
    /// @brief Number of local (x,y) dimensions in a format.
    size_t local_num;
    /// @brief Number of groups in a format.
    size_t group_num;
    /// @brief Dimensions changing order from rare to often.
    std::string order;
    /// @brief Dimensions order for internal storage.
    std::string internal_order;
    /// @brief Block sizes as a vector of pairs of dimension number and block size ordered from rare to often.
    std::vector<std::pair<size_t, int>> block_sizes;
    /// @brief Characters representing batch dimensions in an order.
    static const char* batch_chars() { return "bno"; }
    /// @brief Characters representing feature map/channel dimensions in an order.
    static const char* feature_chars() { return "fic"; }
    /// @brief Characters representing spatial dimensions in an order.
    static const char* spatial_chars() { return "xyzhsw"; }
    /// @brief Characters representing local dimensions in an order.
    static const char* local_chars() { return "kl"; }
    /// @brief Characters representing group dimensions in an order.
    static const char* group_chars() { return "g"; }
    /// @brief Checks if @p c represents batch dimension.
    static bool is_batch_char(char c) { return std::string(batch_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents feature map/channel dimension.
    static bool is_feature_char(char c) { return std::string(feature_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents spatial dimension.
    static bool is_spatial_char(char c) { return std::string(spatial_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents local dimensions.
    static bool is_local_char(char c) { return std::string(local_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents group dimensions.
    static bool is_group_char(char c) { return std::string(group_chars()).find_first_of(c) != std::string::npos; }
};

/// @brief Represents memory formats (orders).
/// @n In CNN most of data is described as 4 dimensional blocks. In Intel(R) clDNN library we describe memory with 4 letters
/// - b - number of blocks in batch. For weights formats: output features - conv, neurons - inner product
/// - f - number of feature maps, features or channels. For weights formats: input features - conv, inputs, inner product
/// - x - spatial, width
/// - y - spatial, height
/// /n
/// For explanation how each format type is implemented in memory we will use naming shown bellow (b=2,f=3,y=3,x=3):
struct format {
    enum type : int32_t {
        // Data formats
        bfyx,                                   ///< the most common format for activations in clDNN.
        bfzyx,                                  ///< format for 5d data tensors
        bfwzyx,                                 ///< batch, feature, 4D spatial
        yxfb,                                   ///< batch first, feature and than spatials
        byxf,                                   ///< used in bitmaps, input from user i.e b images of RGB format
        fyxb,                                   ///< format not used inside clDNN, but supported in reorder as extension
                                                ///< for user provided formats.
        b_fs_yx_fsv16,                          ///< format used for blocked convolution
        b_fs_yx_fsv32,                          ///< format used for blocked int8 convolution
        b_fs_zyx_fsv16,                         ///< format used for 3D blocked convolution (features blocked by 16)
        b_fs_zyx_fsv32,                         ///< format used for blocked int8 3d convolution
        bs_fs_zyx_bsv16_fsv16,                  ///< format used for 3D blocked convolution (batch and features blocked by 16)
        bs_fs_yx_bsv16_fsv16,                   ///< format used for 2D blocked convolution (batch and features blocked by 16)
        bs_fs_yx_bsv4_fsv4,                     ///< format used for 2D blocked convolution (batch and features blocked by 4)
        bs_fs_yx_bsv4_fsv2,                     ///< format used for 2D blocked convolution (batch blocked by 4, features blocked by 2)
        bs_fs_yx_bsv32_fsv32,                   ///< format used for big batches (batch and features blocked by 32)
        bs_fs_yx_bsv32_fsv16,                   ///< format used for big batches (batch blocked by 32, features blocked by 16)
        fs_b_yx_fsv32,                          ///< format for input for fp16 primitives
        b_fs_yx_fsv4,                           ///< format for input for IMAD convolutions
        bs_xs_xsv8_bsv8,                        ///< format used only for fully connected weights: bs - batch slice,
                                                ///< xs - x slice, bsv8 - 8 values of single slice.
        bs_xs_xsv8_bsv16,                       ///< format used only for fully connected weights: bs - batch slice,
                                                ///< xs - x slice, bsv16 - 16 values of single slice.
        bs_x_bsv16,                             ///< format used only for fully connected weights fp16 batch=1 : bs - batch slice
                                                ///< (responses slice), bsv16 - 16 values of single batch slice, x - flattened plane of (fyx)
        b_fs_yx_32fp,                           ///< format for data for binary convolutions
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
        os_iyx_osv16,                                 ///< format used only for convolution weights
        o_is_yx_isv16,                                ///< format used only for convolution weights
        os_yxi_osv16,                                 ///< format used only for convolution weights
        os_is_yx_osv16_isv16,                         ///< format used for convolution i8 weights
        os_is_zyx_osv32_isv16,
        os_is_zyx_osv64_isv16,
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
        os_is_yx_isa8_osv8_isv4_swizzled_by_4,        ///< format for weights for MMAD convolution
        os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,   ///< format for weights for MMAD fsv32 convolution
        os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4,  ///< format for weights for MMAD fsv32 convolution
        os_is_yx_osa4_isa8_osv8_isv4,                 ///< format for weights for MMAD fsv32 convolution
        g_os_is_yx_osa4_isa8_osv8_isv4,               ///< format for weights for MMAD fsv32 convolution
        g_os_is_yx_osa4_isa8_osv8_isv2,               ///< format for weights for MMAD fsv32 convolution
        os_is_yx_osa4_isa8_osv8_isv2,                 ///< format for weights for MMAD fsv32 convolution
        is_o_yx_isv32,                                ///< format for weights for 1x1 MMAD convolutions
        is_o32_yx_isv32_swizzled_by_4,                ///< format for weights for 1x1 MMAD convolutions
        os_is_y_x8_osv8_isv4,                         ///< format for weights for 1x1 MMAD convolutions
        os_is_y_x8_osv8_isv4_swizzled_by_4,           ///< format for weights for 1x1 MMAD convolutions
        os_is_yx_osv16_isv4,                          ///< format for weights for IMAD convolutions
        os_is_yx_osv8_isv4,                           ///< format used for convolution i8 weights
        os_is_yx_osv8_isv2,                           ///< format used for convolution i8 weights
        os_is_zyx_osv16_isv16,                        ///< format for weights for IMAD convolutions
        os_is_yx_osv32_isv4_swizzled_by_2,            ///< format for weights for IMAD convolutions
        os_is_yx_osv32_isv4,                          ///< format for weights for IMAD convolutions
        os_is_zyx_osv32_isv4,                         ///< format for weights for IMAD convolutions
        os_is_yx_osv32_isv32p,                        ///< format for weights for binary convolutions
        lstm_weights_dio,                             ///< dynamic_lstm, direction,
                                                      ///< than IO (I - input size, O - 4 * hidden_size)
        os_is_osv32_isv32_swizzled_by_4,              ///< format for weights for 1x1 IMAD convolution
        os_iyx_osv32__ai32,
        iy_xs_os_xsv2_osv8__ao32,
        iy_xs_os_xsv2_osv16__ao32,
        i_yxs_os_yxsv2_osv16,
        os_i_yxs_osv4_yxsv4,

        goiyx,                                        ///< format used for weights for 2D convolution
        gioyx,                                        ///< format used for weights for 2D deconvolution
        yxiog,                                        ///< format used for weights for 2D convolution
        gyxio,                                        ///< format used for weights for 2D convolution
        goizyx,                                       ///< format used for weights for 3D convolution
        giozyx,                                       ///< format used for weights for 3D deconvolution
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
        any        = -1
    };

    /// @brief Get format traits for particular @p format::type
    static const format_traits& traits(type fmt) {
        static const std::map<type, format_traits> traits {
                // B - number of Batch dimensions
                // F - number of Feature dimensions
                // S - number of Spatial dimensions
                // L - number of Local dimensions
                // G - number of Group dimensions
                // Order - dims changing order from rare to often
                // Inner order - dims order for internal storage in _sizes array
                // Block sizes - vector of pairs of dimension number (by inner order) and block size ordered from rare to often
                // Format                  B  F  S  L  G  Order  Inner order  Block sizes
                { yxfb,                  { 1, 1, 2, 0, 0, "yxfb",   "bfxy?",  {}}},
                { byxf,                  { 1, 1, 2, 0, 0, "byxf",   "bfxy?",  {}}},
                { bfyx,                  { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {}}},
                { fyxb,                  { 1, 1, 2, 0, 0, "fyxb",   "bfxy?",  {}}},
                { b_fs_yx_fsv16,         { 1, 1, 2, 0, 0, "bfyx",   "bfxy",   {{1, 16}}}},
                { b_fs_yx_fsv32,         { 1, 1, 2, 0, 0, "bfyx",   "bfxy",   {{1, 32}}}},
                { b_fs_zyx_fsv32,        { 1, 1, 3, 0, 0, "bfzyx",  "bfxyz",  {{1, 32}}}},
                { bs_xs_xsv8_bsv8,       { 1, 1, 1, 0, 0, "bx",     "b?x??",  {{2, 8}, {0, 8}}}},
                { bs_xs_xsv8_bsv16,      { 1, 1, 1, 0, 0, "bx",     "b?x??",  {{2, 8}, {0, 16}}}},
                { bs_x_bsv16,            { 1, 1, 1, 0, 0, "bx",     "b?x??",  {{0, 16}}}},
                { winograd_2x3_s1_data,  { 1, 1, 2, 0, 0, "bxyf",   "bfxy?",  {}}},
                { b_fs_yx_fsv4,          { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {{1, 4}}}},
                { bfzyx,                 { 1, 1, 3, 0, 0, "bfzyx",  "bfxyz",  {}}},
                { bfwzyx,                { 1, 1, 4, 0, 0, "bfwzyx", "bfxyzw", {}}},
                { fs_b_yx_fsv32,         { 1, 1, 2, 0, 0, "fbyx",   "bfxy?",  {{1, 32}}}},
                { b_fs_yx_32fp,          { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {}}},
                { b_fs_zyx_fsv16,        { 1, 1, 3, 0, 0, "bfzyx",  "bfxyz",  {{1, 16}}}},
                { bs_fs_zyx_bsv16_fsv16, { 1, 1, 3, 0, 0, "bfzyx",  "bfxyz",  {{0, 16 }, {1, 16}}}},
                { bs_fs_yx_bsv16_fsv16,  { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {{0, 16 }, {1, 16}}}},
                { bs_fs_yx_bsv4_fsv4,    { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {{0, 4 }, {1, 4}}}},
                { bs_fs_yx_bsv4_fsv2,    { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {{0, 4 }, {1, 2}}}},
                { bs_fs_yx_bsv32_fsv32,  { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {{0, 32 }, {1, 32}}}},
                { bs_fs_yx_bsv32_fsv16,  { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {{0, 32 }, {1, 16}}}},
                { nv12,                  { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {}}},
                { image_2d_rgba,         { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {}}},

                { oiyx,                                        { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {}}},
                { ioyx,                                        { 1, 1, 2, 0, 0, "ioyx",   "oixy",       {}}},
                { iyxo,                                        { 1, 1, 2, 0, 0, "iyxo",   "oixy",       {}}},
                { yxio,                                        { 1, 1, 2, 0, 0, "yxio",   "oixy?",      {}}},
                { oizyx,                                       { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {}}},
                { iozyx,                                       { 1, 1, 3, 0, 0, "iozyx",  "oixyz",      {}}},
                { os_is_yx_isv16_osv16,                        { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {{1, 16}, {0, 16}}}},
                { o_is_yx_isv16,                               { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {{1, 16}}}},
                { os_yxi_osv16,                                { 1, 1, 2, 0, 0, "oyxi",   "oixy?",      {{0, 16}}}},
                { os_iyx_osv16,                                { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {{0, 16}}}},
                { os_iyx_osv32,                                { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {{0, 32}}}},
                { os_iyx_osv64,                                { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {{0, 64}}}},
                { winograd_2x3_s1_weights,                     { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { winograd_2x3_s1_fused_weights,               { 1, 1, 2, 0, 0, "xyio",   "oixy?",      {}}},
                { winograd_6x3_s1_fused_weights,               { 1, 1, 2, 0, 0, "xyio",   "oixy?",      {}}},
                { image_2d_weights_winograd_6x3_s1_fbxyb,      { 1, 1, 2, 0, 0, "xyio",   "oixy?",      {}}},
                { image_2d_weights_winograd_6x3_s1_xfbyb,      { 1, 1, 2, 0, 0, "xyio",   "oixy?",      {}}},
                { image_2d_weights_c4_fyx_b,                   { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { image_2d_weights_c1_b_fyx,                   { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { lstm_weights_dio,                            { 1, 1, 2, 0, 0, "oixy",   "oixy?",      {}}},
                { os_is_yx_isa8_osv8_isv4,                     { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { os_is_yx_isa8_osv16_isv4,                    { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { os_is_yx_isa8_osv8_isv4_swizzled_by_4,       { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { os_is_yx_osa4_isa8_osv8_isv4,                { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {{0, 32}, {1, 32}}}},
                { os_is_yx_osa4_isa8_osv8_isv2,                { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {{0, 32}, {1, 16}}}},
                { os_is_zyx_isa8_osv8_isv4,                    { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{1, 8}, {0, 8}, {1, 4}}}},
                { os_is_zyx_isa8_osv16_isv4,                   { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{1, 8}, {0, 16}, {1, 4}}}},
                { os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,  { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {{0, 32}, {1, 32}}}},
                { os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4, { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 32}, {1, 32}}}},
                { is_o_yx_isv32,                               { 1, 1, 2, 0, 0, "oyxi",   "oixy?",      {{1, 32}}}},
                { is_o32_yx_isv32_swizzled_by_4,               { 1, 1, 2, 0, 0, "oyxi",   "oixy?",      {}}},
                { os_is_y_x8_osv8_isv4,                        { 1, 1, 2, 0, 0, "oyxi",   "oixy?",      {}}},
                { os_is_y_x8_osv8_isv4_swizzled_by_4,          { 1, 1, 2, 0, 0, "oyxi",   "oixy?",      {}}},
                { os_is_yx_osv16_isv4,                         { 1, 1, 2, 0, 0, "oixy",   "oixy?",      {{0, 16}, {1, 4}}}},
                { os_is_yx_osv8_isv4,                          { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {{1, 4}, {0, 8}}}},
                { os_is_yx_osv8_isv2,                          { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {{1, 2}, {0, 8}}}},
                { os_is_zyx_osv16_isv16,                       { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 16}, {1, 16}}}},
                { os_is_yx_osv32_isv4_swizzled_by_2,           { 1, 1, 2, 0, 0, "oixy",   "oixy?",      {{0, 32}, {1, 4}}}},
                { os_is_yx_osv32_isv4,                         { 1, 1, 2, 0, 0, "oixy",   "oixy?",      {{0, 32}, {1, 4}}}},
                { os_is_zyx_osv32_isv4,                        { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 32}, {1, 4}}}},
                { os_is_yx_osv32_isv32p,                       { 1, 1, 1, 0, 0, "oixy",   "oixy?",      {}}},
                { os_is_zyx_isv16_osv16,                       { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 16}, {1, 16}}}},
                { is_os_zyx_isv16_osv16,                       { 1, 1, 3, 0, 0, "iozyx",  "oixyz",      {{1, 16}, {0, 16}}}},
                { is_os_yx_isv16_osv16,                        { 1, 1, 2, 0, 0, "ioyx",   "oixyz",      {{1, 16}, {0, 16}}}},
                { os_is_osv32_isv32_swizzled_by_4,             { 1, 1, 0, 0, 0, "oixy",   "oixy?",      {{0, 32}, {1, 32}}}},
                { os_is_zyx_isv8_osv16_isv2,                   { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{1, 8}, {0, 16}, {1, 2}}}},
                { os_zyxi_osv16,                               { 1, 1, 3, 0, 0, "ozyxi",  "oixyz",      {{0, 16}}}},
                { os_is_yx_isv8_osv16_isv2,                    { 1, 1, 2, 0, 0, "oizyx",  "oixyz",      {{1, 8}, {0, 16}, {1, 2}}}},
                { os_is_yx_osv16_isv16,                        { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {{1, 16}, {0, 16}}}},
                { os_is_zyx_osv32_isv16,                       { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 32}, {1, 16}}}},
                { os_is_zyx_osv64_isv16,                       { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 64}, {1, 16}}}},
                { os_iyx_osv32__ai32,                          { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {{0, 32}}}},
                { i_yxs_os_yxsv2_osv16,                        { 1, 1, 2, 0, 0, "iyxo",   "oixy",       {{0, 16}}}},
                { iy_xs_os_xsv2_osv8__ao32,                    { 1, 1, 2, 0, 0, "iyxo",   "oixy",       {{2, 2}, {0, 8}}}},
                { iy_xs_os_xsv2_osv16__ao32,                   { 1, 1, 2, 0, 0, "iyxo",   "oixy",       {{2, 2}, {0, 16}}}},
                { os_i_yxs_osv4_yxsv4,                         { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {{0, 4}}}},

                { goiyx,                                       { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {}}},
                { gioyx,                                       { 1, 1, 2, 0, 1, "gioyx",  "oixy????g",  {}}},
                { goizyx,                                      { 1, 1, 3, 0, 1, "goizyx", "oixyz???g",  {}}},
                { giozyx,                                      { 1, 1, 3, 0, 1, "giozyx", "oixyz???g",  {}}},
                { g_os_iyx_osv16,                              { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{0, 16}}}},
                { g_os_iyx_osv32,                              { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{0, 32}}}},
                { gs_oiyx_gsv16,                               { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{8, 16}}}},
                { gs_oizyx_gsv16,                              { 1, 1, 3, 0, 1, "goizyx", "oixyz???g",  {{8, 16}}}},
                { gs_oiyx_gsv32,                               { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{8, 32}}}},
                { gyxio,                                       { 1, 1, 2, 0, 1, "gyxio",  "oixy????g",  {}}},
                { g_is_os_zyx_isv16_osv16,                     { 1, 1, 3, 0, 1, "giozyx", "oixyz???g",  {{1, 16}, {0, 16}}}},
                { g_is_os_yx_isv16_osv16,                      { 1, 1, 2, 0, 1, "gioyx",  "oixy????g",  {{1, 16}, {0, 16}}}},
                { g_os_is_zyx_isv8_osv16_isv2,                 { 1, 1, 3, 0, 1, "goizyx", "oixyz???g",  {{1, 8}, {0, 16}, {1, 2}}}},
                { g_os_is_yx_isv8_osv16_isv2,                  { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{1, 8}, {0, 16}, {1, 2}}}},
                { g_os_is_zyx_isv16_osv16,                     { 1, 1, 3, 0, 1, "goizyx", "oixyz???g",  {{0, 16}, {1, 16}}}},
                { g_os_is_yx_osv16_isv4,                       { 1, 1, 2, 0, 1, "goixy",  "oixy????g",  {{0, 16}, {1, 4}}}},
                { g_os_is_zyx_osv16_isv16,                     { 1, 1, 3, 0, 1, "goizyx", "oixyz???g",  {{0, 16}, {1, 16}}}},
                { g_os_zyx_is_osv16_isv4,                      { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 16}, {1, 4}}}},
                { g_os_zyx_is_osv16_isv16,                     { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 16}, {1, 16}}}},
                { g_os_zyx_is_osv16_isv32,                     { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 16}, {1, 32}}}},
                { g_os_zyx_is_osv32_isv4,                      { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 32}, {1, 4}}}},
                { g_os_zyx_is_osv32_isv16,                     { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 32}, {1, 16}}}},
                { g_os_zyx_is_osv32_isv32,                     { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 32}, {1, 32}}}},
                { g_os_is_yx_osa4_isa8_osv8_isv4,              { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{0, 32}, {1, 32}}}},
                { g_os_is_yx_osa4_isa8_osv8_isv2,              { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{0, 32}, {1, 16}}}},
                { gs_oi_yxs_gsv4_yxsv4,                        { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{8, 4}}}},
                { gs_oi_yxs_gsv16_yxsv4,                       { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{8, 16}}}},
                { gs_oi_yxs_gsv32_yxsv4,                       { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{8, 32}}}},
                { g_os_is_yx_isv16_osv16,                      { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{1, 16}, {0, 16}}}},
                { gi_yxs_os_yxsv2_osv16,                       { 1, 1, 2, 0, 1, "giyxo",  "oixy????g",  {{0, 16}}}},
                { giy_xs_os_xsv2_osv8__ao32,                   { 1, 1, 2, 0, 1, "giyxo",  "oixy????g",  {{2, 2}, {0, 8}}}},
                { giy_xs_os_xsv2_osv16__ao32,                  { 1, 1, 2, 0, 1, "giyxo",  "oixy????g",  {{2, 2}, {0, 16}}}},
        };
        if (traits.find(fmt) == traits.end()) {
            throw std::runtime_error("[clDNN] Format description is missing in fmt traits");
        }
        return traits.at(fmt);
    }

    /// @brief Returns number of batch dimensions for a @p format.
    static size_t batch_num(type fmt) { return traits(fmt).batch_num; }
    /// @brief Returns number of feature dimensions for a @p format.
    static size_t feature_num(type fmt) { return traits(fmt).feature_num; }
    /// @brief Returns number of spatial dimensions for a @p format.
    static size_t spatial_num(type fmt) { return traits(fmt).spatial_num; }
    /// @brief Returns number of local dimensions for a @p format.
    static size_t local_num(type fmt) { return traits(fmt).local_num; }
    /// @brief Returns number of group dimensions for a @p format.
    static size_t group_num(type fmt) { return traits(fmt).group_num; }
    /// @brief Returns an order of dimensions for a @ format.
    static const std::string& order(type fmt) { return traits(fmt).order; }
    /// @brief Returns an internal orders of dimensions for a @p format.
    static const std::string& internal_order(type fmt) { return traits(fmt).internal_order; }
    /// @brief Returns block sizes for @p format.
    static const std::vector<std::pair<size_t, int>>& block_sizes(type fmt) { return traits(fmt).block_sizes; }
    /// @brief Returns number of dimensions contained within a @p format
    static size_t dimension(type fmt) { return order(fmt).size(); }
    /// @brief Checks if @p format is a winograd format
    static bool is_winograd(type fmt) {
        return (fmt == winograd_2x3_s1_data ||
                fmt == winograd_2x3_s1_weights ||
                fmt == winograd_2x3_s1_fused_weights ||
                fmt == winograd_6x3_s1_fused_weights ||
                fmt == image_2d_weights_winograd_6x3_s1_fbxyb ||
                fmt == image_2d_weights_winograd_6x3_s1_xfbyb); }
    /// @brief Checks if @p format is of image2d type
    static bool is_image_2d(type fmt) {
        return (fmt == image_2d_weights_c4_fyx_b ||
                fmt == image_2d_weights_c1_b_fyx ||
                fmt == image_2d_weights_winograd_6x3_s1_fbxyb ||
                fmt == image_2d_weights_winograd_6x3_s1_xfbyb ||
                fmt == nv12 ||
                fmt == image_2d_rgba);
    }
    /// @brief Checks if @p format is weights format
    static bool is_weights_format(type fmt) {
        const auto internal_order = traits(fmt).internal_order;
        const auto weights_chars = { "o", "i" };
        for (const auto& c : weights_chars) {
            if (internal_order.find_first_of(c) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
    /// @brief Checks if @p format is simple data format
    static bool is_simple_data_format(type fmt) {
        return (fmt == yxfb || fmt == byxf ||
                fmt == bfyx || fmt == fyxb ||
                fmt == bfzyx || fmt == bfwzyx);
    }
    /// @brief Checks if @p format is of grouped type
    static bool is_grouped(type fmt) { return group_num(fmt) != 0; }
    /// @brief Checks if @p format is of image type
    static bool is_image(type fmt) { return (is_image_2d(fmt)); }
    /// @brief Checks if @p format is blocked format
    static bool is_blocked(type fmt) { return !(block_sizes(fmt).empty()); }
    /// @brief Checks if @p format is nv12 format
    static bool is_nv12(type fmt) { return (fmt == nv12); }

    /// @brief Returns number of batch dimensions.
    size_t batch_num() const { return traits(value).batch_num; }
    /// @brief Returns number of feature dimensions.
    size_t feature_num() const { return traits(value).feature_num; }
    /// @brief Returns number of spatial dimensions.
    size_t spatial_num() const { return traits(value).spatial_num; }
    /// @brief Returns number of local dimensions.
    size_t local_num() const { return traits(value).local_num; }
    /// @brief Returns number of group dimensions.
    size_t group_num() const { return traits(value).group_num; }
    /// @brief Returns an order of dimensions in form of string.
    const std::string& order() const { return traits(value).order; }
    /// @brief Returns an internal orders of dimensions form of string.
    const std::string& internal_order() const { return traits(value).internal_order; }
    /// @brief Returns block sizes as vector of pairs of dimension and block size for that dimension.
    const std::vector<std::pair<size_t, int>>& block_sizes() const { return traits(value).block_sizes; }
    /// @brief Returns number of dimensions contained within this format
    size_t dimension() const { return order(value).size(); }
    /// @brief Checks if @p format is a winograd format
    bool is_winograd() const { return is_winograd(value); }
    /// @brief Checks if @p format is of image 2d type
    bool is_image_2d() const { return is_image_2d(value); }
    /// @brief Checks if @p format is of image type
    bool is_image() const { return is_image(value); }
    /// @brief Checks if @p format is blocked format
    bool is_blocked() { return is_blocked(value); }
    /// @brief Checks if @p format is a nv12 format
    bool is_nv12() const { return is_nv12(value); }

    /// @brief Transforms dimension from internal order to external order
    size_t internal_to_external(size_t idx) const {
        auto index = order().find_first_of(internal_order()[idx]);
        if (index == std::string::npos)
            throw std::invalid_argument("Internal dimension index does not map to external index.");
        return index;
    }

    type value;
    /// @brief Implicit conversion from format::type.
    constexpr format(type t) : value(t) {}
    /// @brief Implicit conversion to format::type.
    constexpr operator type() const { return value; }
};

constexpr int32_t tensor_batch_dim_max = 1;
constexpr int32_t tensor_feature_dim_max = 1;
constexpr int32_t tensor_spatial_dim_max = 4;
constexpr int32_t tensor_local_dim_max = 2;
constexpr int32_t tensor_group_dim_max = 1;
constexpr int32_t tensor_dim_max = tensor_batch_dim_max + tensor_feature_dim_max + tensor_spatial_dim_max + tensor_local_dim_max + tensor_group_dim_max;

struct tensor;

/// @brief Helper structs used in tensor constructor with dim_vec_kinds
namespace details {
/// @brief enum class that represent dimension kinds
enum class dim_vec_kind {
    batch,
    feature,
    spatial,
    local,
    group
};

/// @brief template class with max_dimensionalities and dimension offset for dimension kinds
template <dim_vec_kind Kind>
struct dim_vec_limits {
    static_assert(meta::always_false_ty_val<dim_vec_kind, Kind>::value, "Limits are undefined for selected value of dim_vec_kind.");
};

template <>
struct dim_vec_limits<dim_vec_kind::batch> {
    static constexpr int32_t max_dimentionality = tensor_batch_dim_max;
    static constexpr int32_t dim_offset = 0;
};

template <>
struct dim_vec_limits<dim_vec_kind::feature> {
    static constexpr int32_t max_dimentionality = tensor_feature_dim_max;
    static constexpr int32_t dim_offset = tensor_batch_dim_max;
};

template <>
struct dim_vec_limits<dim_vec_kind::spatial> {
    static constexpr int32_t max_dimentionality = tensor_spatial_dim_max;
    static constexpr int32_t dim_offset = tensor_batch_dim_max + tensor_feature_dim_max;
};

template <>
struct dim_vec_limits<dim_vec_kind::local> {
    static constexpr int32_t max_dimentionality = tensor_local_dim_max;
    static constexpr int32_t dim_offset = tensor_batch_dim_max + tensor_feature_dim_max + tensor_spatial_dim_max;
};

template <>
struct dim_vec_limits<dim_vec_kind::group> {
    static constexpr int32_t max_dimentionality = tensor_group_dim_max;
    static constexpr int32_t dim_offset = tensor_batch_dim_max + tensor_feature_dim_max + tensor_spatial_dim_max + tensor_local_dim_max;
};

/// @brief Template class used in tensor constructor using dim_vec_kinds
template <dim_vec_kind Kind>
class dim_vec_kind_init {
public:
    static constexpr auto _max_dimensionality = dim_vec_limits<Kind>::max_dimentionality;
    static constexpr auto _dimOffset = dim_vec_limits<Kind>::dim_offset;

    template <typename... DimTys>
    explicit dim_vec_kind_init(DimTys&&... values)
        : _sizes{int32_t(std::forward<DimTys>(values))...}, _dimSize(sizeof...(DimTys)) {
    }

    void init_tensor_values(cldnn::tensor& t);

    int32_t _sizes[_max_dimensionality];
    int32_t _dimSize;
};
}  // namespace details

template <typename... InitTys>
details::dim_vec_kind_init<details::dim_vec_kind::batch> batch(InitTys&&... inits) {
    return details::dim_vec_kind_init<details::dim_vec_kind::batch>(std::forward<InitTys>(inits)...);
}

template <typename... InitTys>
details::dim_vec_kind_init<details::dim_vec_kind::feature> feature(InitTys&&... inits) {
    return details::dim_vec_kind_init<details::dim_vec_kind::feature>(std::forward<InitTys>(inits)...);
}

template <typename... InitTys>
details::dim_vec_kind_init<details::dim_vec_kind::spatial> spatial(InitTys&&... inits) {
    return details::dim_vec_kind_init<details::dim_vec_kind::spatial>(std::forward<InitTys>(inits)...);
}

template <typename... InitTys>
details::dim_vec_kind_init<details::dim_vec_kind::local> local(InitTys&&... inits) {
    return details::dim_vec_kind_init<details::dim_vec_kind::local>(std::forward<InitTys>(inits)...);
}

template <typename... InitTys>
details::dim_vec_kind_init<details::dim_vec_kind::group> group(InitTys&&... inits) {
    return details::dim_vec_kind_init<details::dim_vec_kind::group>(std::forward<InitTys>(inits)...);
}

/// @brief N-dimensional vector. Mostly used to represent memory size.
struct tensor {
    friend class details::dim_vec_kind_init<details::dim_vec_kind::batch>;
    friend class details::dim_vec_kind_init<details::dim_vec_kind::feature>;
    friend class details::dim_vec_kind_init<details::dim_vec_kind::spatial>;
    friend class details::dim_vec_kind_init<details::dim_vec_kind::local>;
    friend class details::dim_vec_kind_init<details::dim_vec_kind::group>;

    typedef int32_t value_type;  ///< Values type stored in tensor.
    // TODO find the way to prevent direct change of following fields.
    mutable_array_ref<value_type> raw;      ///< Raw representation of all dimensions.
    mutable_array_ref<value_type> batch;    ///< Batch dimensions.
    mutable_array_ref<value_type> feature;  ///< Feature maps.
    mutable_array_ref<value_type> spatial;  ///< Spatial dimensions.
    mutable_array_ref<value_type> local;    ///< Local dimensions.
    mutable_array_ref<value_type> group;    ///< Group dimensions.

private:
    value_type _sizes[tensor_dim_max];
    value_type _dimOffset;
    value_type _dimSize;

public:
    explicit tensor(value_type default_size = 0) :
        raw(_sizes, tensor_dim_max),
        batch(_sizes, tensor_batch_dim_max),
        feature(_sizes + tensor_batch_dim_max, tensor_feature_dim_max),
        spatial(_sizes + tensor_batch_dim_max + tensor_feature_dim_max, tensor_spatial_dim_max),
        local(_sizes + tensor_batch_dim_max + tensor_feature_dim_max + tensor_spatial_dim_max, tensor_local_dim_max),
        group(_sizes + tensor_batch_dim_max + tensor_feature_dim_max + tensor_spatial_dim_max + tensor_local_dim_max, tensor_group_dim_max) {
        std::fill_n(_sizes, tensor_dim_max, default_size);
    }

    /// @brief Constructs tensor.
    /// @param[in] kind_inits Dimensions defined using dim_vec_kind. If dimension is not provided it is set to 1.
    /// @details Example:
    /*! @code
    *
    tensor my_tensor(batch(2), spatial(5, 6));   // y=6, x=5, b=2, f - not set
    cout << my_tensor.batch[0] << endl;           // 2
    cout << my_tensor.feature[0] << endl;         // 1 - default_size
    cout << "x=" << my_tensor.spatial[0] << endl; // x=5
    cout << "y=" << my_tensor.spatial[1] << endl; // y=6
    *
    * @endcode
    */
    template <typename... KindInitTys,
              typename = typename std::enable_if<
                  meta::all<
                      meta::is_any_of<KindInitTys,
                                      cldnn::details::dim_vec_kind_init<details::dim_vec_kind::batch>,
                                      cldnn::details::dim_vec_kind_init<details::dim_vec_kind::feature>,
                                      cldnn::details::dim_vec_kind_init<details::dim_vec_kind::spatial>,
                                      cldnn::details::dim_vec_kind_init<details::dim_vec_kind::group>>::value...>::value,
                  void>::type>
    explicit tensor(KindInitTys&&... kind_inits)
        : tensor(1) {
        assign_inits(std::forward<KindInitTys>(kind_inits)...);
    }

    /// @brief Constructs @p tensor.
    /// @details Example:
    /*! @code
     *
       tensor my_tensor( 2, 3, 4, 5 );   // b=2, f=3, x=4, y=5
       cout << my_tensor.batch[0] << endl;           // 2
       cout << my_tensor.feature[0] << endl;         // 3
       cout << "x=" << my_tensor.spatial[0] << endl; // x=4
       cout << "y=" << my_tensor.spatial[1] << endl; // y=5
     *
     * @endcode
     */
    tensor(value_type batch_num, value_type feature_num, value_type x, value_type y)
        : tensor(1) {
        _sizes[0] = batch_num;
        _sizes[tensor_batch_dim_max] = feature_num;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max] = x;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 1] = y;
        if (batch_num == 0 && feature_num == 0 && x == 0 && y == 0)
            _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 2] = 0;
    }

    /// @brief Constructs @p tensor.
    /// @details Example:
    /*! @code
    *
    tensor my_tensor( 2, 3, 4, 5, 6 );   // b=2, f=3, x=4, y=5, z=6
    cout << my_tensor.batch[0] << endl;           // 2
    cout << my_tensor.feature[0] << endl;         // 3
    cout << "x=" << my_tensor.spatial[0] << endl; // x=4
    cout << "y=" << my_tensor.spatial[1] << endl; // y=5
    cout << "z=" << my_tensor.spatial[2] << endl; // z=6
    *
    * @endcode
    */
    tensor(value_type batch_num, value_type feature_num, value_type x, value_type y, value_type z)
        : tensor(1) {
        _sizes[0] = batch_num;
        _sizes[tensor_batch_dim_max] = feature_num;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max] = x;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 1] = y;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 2] = z;
    }

    /// @brief Constructs @p tensor.
    /// @details Example:
    /*! @code
    *
    tensor my_tensor( 2, 3, 4, 5, 6, 7 );   // b=2, f=3, x=4, y=5, z=6, w=7
    cout << my_tensor.batch[0] << endl;           // 2
    cout << my_tensor.feature[0] << endl;         // 3
    cout << "x=" << my_tensor.spatial[0] << endl; // x=4
    cout << "y=" << my_tensor.spatial[1] << endl; // y=5
    cout << "z=" << my_tensor.spatial[2] << endl; // z=6
    cout << "w=" << my_tensor.spatial[3] << endl; // w=7
    *
    * @endcode
    */
    tensor(value_type batch_num, value_type feature_num, value_type x, value_type y, value_type z, value_type w)
        : tensor(1) {
        _sizes[0] = batch_num;
        _sizes[tensor_batch_dim_max] = feature_num;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max] = x;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 1] = y;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 2] = z;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 3] = w;
    }

    /// @brief Constructs @p tensor using vector of sizes.
    /// @param[in] sizes dimensions need to be provided in the following order {batch, feature, spatial_x, spatial_y [, spatial_z] }.
    /// @param[in] default_size default_size for tensor dimensions.
    /// @details Example:
    /*! @code
     *
       tensor my_tensor = { 2, 3, 4, 5 };   // b=2, f=3, x=4, y=5
       cout << my_tensor.batch[0] << endl;           // 2
       cout << my_tensor.feature[0] << endl;         // 3
       cout << "x=" << my_tensor.spatial[0] << endl; // x=4
       cout << "y=" << my_tensor.spatial[1] << endl; // y=5
     *
     * @endcode
     */
    explicit tensor(const std::vector<value_type>& sizes, value_type default_size = 1)
        : tensor(default_size) {
        int max_size = std::min(static_cast<int>(sizes.size()), tensor_dim_max);
        for (int i = 0; i < max_size; i++)
            _sizes[i] = sizes[i];
    }

    tensor(format fmt, const std::vector<value_type>& sizes, value_type default_size = 1)
        : tensor(default_size) {
        auto in_order = fmt.order();
        auto out_order = fmt.internal_order();
        if (in_order.size() != sizes.size())
            throw std::invalid_argument("The count of values passed to initialize tensor does not match passed format.");

        for (size_t out_idx = 0; out_idx < out_order.size(); ++out_idx) {
            auto channel = out_order[out_idx];
            if (channel == '?')
                continue;

            auto in_idx = in_order.find(channel);
            if (in_idx == in_order.npos)
                throw std::runtime_error("Internal order of a format contains channel which does not appear in external order.");

            _sizes[out_idx] = sizes[in_idx];
        }
    }

    /// @brief Copy construction.
    tensor(const tensor& other)
        : tensor(0) {
        std::copy_n(other._sizes, tensor_dim_max, _sizes);
    }

    /// @brief Copy assignment.
    tensor& operator=(const tensor& other) {
        if (this == &other)
            return *this;
        std::copy_n(other._sizes, tensor_dim_max, _sizes);
        return *this;
    }

    friend bool operator==(const tensor& lhs, const tensor& rhs) {
        return lhs.raw.size() == rhs.raw.size() && std::equal(lhs.raw.begin(), lhs.raw.end(), rhs.raw.begin());
    }

    friend bool operator!=(const tensor& lhs, const tensor& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator<(const tensor& lhs, const tensor& rhs) {
        if (lhs.raw.size() != rhs.raw.size())
            return lhs.raw.size() < rhs.raw.size();
        for (size_t i = 0; i < lhs.raw.size(); ++i) {
            if (lhs.raw[i] < rhs.raw[i])
                return true;
            if (rhs.raw[i] < lhs.raw[i])
                return false;
        }

        return false;
    }

    friend std::ostream& operator<<(std::ostream& os, const tensor& tensor) {
        os << tensor.to_string();
        return os;
    }

    std::string to_string() const {
        std::stringstream out;
        const char* delim = "";

        out << "[b:";
        for (size_t i = 0; i < batch.size(); ++i) {
            out << delim << batch[i];
            delim = ",";
        }
        delim = "";

        out << ", f:";
        for (size_t i = 0; i < feature.size(); ++i) {
            out << delim << feature[i];
            delim = ",";
        }

        std::vector<std::string> spatial_dim_names = {", x", ", y", ", z", ", w"};
        for (size_t i = 0; i < spatial.size(); ++i) {
            out << spatial_dim_names[i] << ":" << spatial[i];
        }

        out << ", g:";
        for (size_t i = 0; i < group.size(); ++i) {
            out << group[i];
        }
        out << "]";

        return out.str();
    }

    /// @brief Returns a tensor with all negated elements.
    tensor negate() const {
        auto result = *this;
        for (size_t i = 0; i < tensor_dim_max; i++) {
            result._sizes[i] = -_sizes[i];
        }
        return result;
    }

    /// @brief Returns a tensor with all elements multilied to @p multiplier.
    tensor mul(value_type multiplier) const {
        auto result = *this;
        for (size_t i = 0; i < tensor_dim_max; i++) {
            result._sizes[i] *= multiplier;
        }
        return result;
    }

    /// @brief Returns a tensor with all elements divided by @p divider.
    tensor div(value_type divider) const {
        auto result = *this;
        for (size_t i = 0; i < tensor_dim_max; i++) {
            result._sizes[i] /= divider;
        }
        return result;
    }

    /// @brief Returns a tensor with all elements added by appropriate elements of @p rhs
    tensor add(const tensor& rhs) const {
        auto result = *this;
        for (size_t i = 0; i < tensor_dim_max; i++) {
            result._sizes[i] += rhs._sizes[i];
        }
        return result;
    }

    /// @brief Returns a tensor with all elements subtracted by appropriate elements of @p rhs
    tensor sub(const tensor& rhs) const {
        return add(rhs.negate());
    }

    /// @brief Assign and add
    tensor& operator+=(const tensor& rhs) {
        for (size_t i = 0; i < tensor_dim_max; i++)
            _sizes[i] += rhs._sizes[i];
        return *this;
    }

    /// @brief Assign and subtract
    tensor& operator-=(const tensor& rhs) {
        for (size_t i = 0; i < tensor_dim_max; i++)
            _sizes[i] -= rhs._sizes[i];
        return *this;
    }

    /// @brief Returns a vector of tensors values, ordered regarding to @p format.
    std::vector<value_type> sizes(cldnn::format fmt) const {
        auto output_order = fmt.order();
        auto internal_order = fmt.internal_order();
        std::vector<value_type> sizes(output_order.size(), 0);

        for (size_t i = 0; i < sizes.size(); ++i) {
            auto c = output_order[i];
            auto pos = internal_order.find(c);
            if (pos == internal_order.npos)
                throw std::domain_error(std::string("Unknown coord type: ") + c);

            sizes[i] = _sizes[pos];
        }

        return sizes;
    }

    /// @brief Returns a vector of tensors values, ordered batch, feature, spatial_x, spatial_y.
    std::vector<value_type> sizes() const {
        std::vector<value_type> sizes(sizeof(_sizes) / sizeof(_sizes[0]), 0);
        for (size_t i = 0; i < sizes.size(); ++i)
            sizes[i] = _sizes[i];
        return sizes;
    }

    /// @brief Returns tensor elements count calculated as multiplication of all elements.
    size_t count() const {
        return std::accumulate(
            raw.begin(),
            raw.end(),
            static_cast<size_t>(1),
            std::multiplies<size_t>());
    }

    /// @brief Returns new tensor based on current but transformed to new @p format.
    /// @param[in] new_fmt Format of new tensor.
    /// @param[in] default_size Default element values for positions not defined by current format.
    /// @details Example:
    /*!
     * @code
       tensor my_tensor({ 2, 3, 4, 5 });
       auto my_sizes = my_tensor.sizes();
       cout << "dims_num=" << my_sizes.size() << endl; // dims_num=2
       cout << "b=" << my_sizes[0] << endl;            // b=2
       cout << "f=" << my_sizes[1] << endl;            // f=3
       cout << "x=" << my_sizes[2] << endl;            // x=5
       cout << "y=" << my_sizes[3] << endl;            // y=4
       auto new_tensor = my_tensor.transform(format::yxfb, 10);
       auto new_sizes = new_tensor.sizes();
       cout << "new_num=" << new_sizes.size() << endl;   // new_num=4
       for(auto dim : new_sizes) cout << " " << dim;     //  5 4 3 2
       cout << endl;
       * @endcode
     */
    tensor transform(cldnn::format new_fmt, value_type default_size) const {
        cldnn::format format = cldnn::format::bfwzyx;
        auto val_order = format.internal_order();
        auto new_order = new_fmt.internal_order();
        std::vector<value_type> old_sizes = sizes();
        std::vector<value_type> new_sizes(old_sizes.size(), default_size);
        auto tmp = 1;
        auto tmp_z = 1;
        auto tmp_w = 1;
        for (size_t i = 0; i < format.order().size(); i++) {
            auto c = val_order[i];
            // skip f and y, z for the formats that do not have it
            if (((new_fmt == format::bs_xs_xsv8_bsv8) ||
                 (new_fmt == format::bs_xs_xsv8_bsv16) ||
                 (new_fmt == format::bs_x_bsv16)) &&
                ((c == 'f') ||
                 (c == 'y') ||
                 (c == 'z') ||
                 (c == 'w'))) {
                if (new_order[i] == '?')
                    new_sizes[i] = default_size;

                tmp *= old_sizes[i];
                continue;
            }

            // skip z for the formats that do not have it
            if (((new_fmt != format::bfzyx && new_fmt != format::b_fs_zyx_fsv16 && new_fmt != format::b_fs_zyx_fsv32 &&
                  new_fmt != format::bfwzyx && new_fmt != format::bs_fs_zyx_bsv16_fsv16)) && (c == 'z')) {
                if (new_order[i] == '?')
                    new_sizes[i] = default_size;

                tmp_z *= old_sizes[i];
                continue;
            }

            if (new_fmt != format::bfwzyx && c == 'w') {
                if (new_order[i] == '?')
                    new_sizes[i] = default_size;

                if (new_fmt == format::bfzyx || new_fmt == format::b_fs_zyx_fsv16 ||
                    new_fmt == format::bs_fs_zyx_bsv16_fsv16 || new_fmt == format::b_fs_zyx_fsv32)
                    tmp_w *= old_sizes[i];
                else
                    tmp_z *= old_sizes[i];
                continue;
            }

            auto new_pos = new_order.find(c);
            if (new_pos == std::string::npos)
                throw std::invalid_argument("cannot convert to new format");
            new_sizes[new_pos] = old_sizes[i];
        }

        // in case of formats with smaller number of dimensions than input, flatten is performed below
        if (tmp != 1 || tmp_z != 1 || tmp_w != 1) {
            for (size_t i = 0; i < format.order().size(); i++) {
                auto c = val_order[i];
                if (c == 'x') {
                    auto new_pos = new_order.find(c);
                    new_sizes[new_pos] *= tmp;
                }
                if (c == 'y') {
                    auto new_pos = new_order.find(c);
                    if (new_pos != std::string::npos)
                        new_sizes[new_pos] *= tmp_z;
                }
                if (c == 'z') {
                    auto new_pos = new_order.find(c);
                    if (new_pos != std::string::npos)
                        new_sizes[new_pos] *= tmp_w;
                }
            }
        }

        tensor sizes { new_sizes };
        return sizes;
    }

    /// @brief Calculates linear offset for given @p coord within current tensor.
    /// @param coord The coordinate within current tensor.
    size_t get_linear_offset(const tensor& coord, cldnn::format fmt) const {
        auto my_sizes = this->sizes(fmt);
        auto adjusted_coords = coord.sizes(fmt);

        // Extend N-dimensional format with B blocked dimensions to (N+B) sizes
        for (const auto& block : fmt.block_sizes()) {
            auto block_axis = block.first;
            auto block_size = block.second;
            auto external_axis = fmt.internal_to_external(block_axis);

            my_sizes.push_back(block_size);
            my_sizes[external_axis] = ceil_div(my_sizes[external_axis], block_size);

            adjusted_coords.push_back(adjusted_coords[external_axis] % block_size);
            adjusted_coords[external_axis] /= block_size;
        }

        if (fmt == cldnn::format::os_is_yx_isa8_osv8_isv4 &&  // TODO Fix offsets calculation for formats below
                   !(is_aligned_to(my_sizes[0], 8)) &&
                   !(is_aligned_to(my_sizes[1], 32))) {
            my_sizes[0] = align_to(my_sizes[0], 8);
            my_sizes[1] = align_to(my_sizes[1], 32);
            adjusted_coords[0] = align_to(adjusted_coords[0], 8);
            adjusted_coords[1] = align_to(adjusted_coords[1], 32);
        } else if (fmt == cldnn::format::os_is_yx_isa8_osv16_isv4 &&
                   !(is_aligned_to(my_sizes[0], 16)) &&
                   !(is_aligned_to(my_sizes[1], 32))) {
            my_sizes[0] = align_to(my_sizes[0], 16);
            my_sizes[1] = align_to(my_sizes[1], 32);
            adjusted_coords[0] = align_to(adjusted_coords[0], 16);
            adjusted_coords[1] = align_to(adjusted_coords[1], 32);
        } else if (fmt == cldnn::format::os_is_yx_isa8_osv8_isv4_swizzled_by_4 && !(is_aligned_to(my_sizes[0], 32)) && !(is_aligned_to(my_sizes[1], 32))) {
            my_sizes[0] = align_to(my_sizes[0], 32);
            my_sizes[1] = align_to(my_sizes[1], 32);
            adjusted_coords[0] = align_to(adjusted_coords[0], 32);
            adjusted_coords[1] = align_to(adjusted_coords[1], 32);
        } else if (fmt == cldnn::format::is_o32_yx_isv32_swizzled_by_4 && (!is_aligned_to(my_sizes[1], 32) || !is_aligned_to(my_sizes[0], 32))) {
            my_sizes[0] = align_to(my_sizes[0], 32);
            my_sizes[3] = align_to(my_sizes[3], 32);
            adjusted_coords[0] = align_to(adjusted_coords[0], 32);
            adjusted_coords[3] = align_to(adjusted_coords[3], 32);
        } else if (fmt == cldnn::format::os_is_y_x8_osv8_isv4 || fmt == cldnn::format::os_is_yx_isa8_osv8_isv4_swizzled_by_4) {
            my_sizes[1] = align_to(my_sizes[1], 4);
            my_sizes[0] = align_to(my_sizes[0], 8);
            my_sizes[2] = align_to(my_sizes[2], 8);
        } else if (fmt == cldnn::format::gs_oi_yxs_gsv4_yxsv4 || fmt == cldnn::format::gs_oi_yxs_gsv16_yxsv4 || fmt == cldnn::format::gs_oi_yxs_gsv32_yxsv4) {
            const auto yxsv = 4;
            const auto flat_xy = adjusted_coords[4] + adjusted_coords[3] * my_sizes[4];

            my_sizes.push_back(yxsv);
            my_sizes[4] = ceil_div(my_sizes[3] * my_sizes[4], yxsv);
            my_sizes[3] = 1;

            adjusted_coords.push_back(flat_xy % yxsv);
            adjusted_coords[4] = flat_xy / yxsv;
            adjusted_coords[3] = 0;
        } else if (fmt == cldnn::format::os_iyx_osv32__ai32 && !is_aligned_to(my_sizes[1], 32)) {
            my_sizes[1] = align_to(my_sizes[1], 32);
        } else if ((fmt == cldnn::format::iy_xs_os_xsv2_osv8__ao32 || fmt == cldnn::format::iy_xs_os_xsv2_osv16__ao32) && !is_aligned_to(my_sizes[3], 32)) {
            my_sizes[3] = align_to(my_sizes[3], 32);
        } else if (fmt == cldnn::format::i_yxs_os_yxsv2_osv16 || fmt == cldnn::format::gi_yxs_os_yxsv2_osv16) {
            const auto yxsv = 2;
            auto flat_xy = adjusted_coords[2] + adjusted_coords[1] * my_sizes[2];

            my_sizes.insert(std::prev(my_sizes.end()), yxsv);
            my_sizes[2] = ceil_div(my_sizes[1] * my_sizes[2], yxsv);
            my_sizes[1] = 1;

            adjusted_coords.insert(std::prev(adjusted_coords.end()), flat_xy % yxsv);
            adjusted_coords[2] = flat_xy / yxsv;
            adjusted_coords[1] = 0;
        } else if (fmt == cldnn::format::os_i_yxs_osv4_yxsv4) {
            const auto yxsv = 4;
            const auto flat_xy = adjusted_coords[3] + adjusted_coords[2] * my_sizes[3];

            my_sizes.push_back(yxsv);
            my_sizes[3] = ceil_div(my_sizes[2] * my_sizes[3], yxsv);
            my_sizes[2] = 1;

            adjusted_coords.push_back(flat_xy % yxsv);
            adjusted_coords[3] = flat_xy / yxsv;
            adjusted_coords[2] = 0;
        } else if ((fmt == cldnn::format::giy_xs_os_xsv2_osv8__ao32 || fmt == cldnn::format::giy_xs_os_xsv2_osv16__ao32) && !is_aligned_to(my_sizes[3], 32)) {
            my_sizes[4] = align_to(my_sizes[4], 32);
        }

        assert(my_sizes.size() == adjusted_coords.size());
        assert(adjusted_coords.size() > 0);

        size_t offset = adjusted_coords[0];
        for (size_t i = 1; i < adjusted_coords.size(); i++) {
            offset = offset * my_sizes[i] + adjusted_coords[i];
        }
        return offset;
    }

    /// @brief Returns a tensor containing values maximum from @p lhs and @p rhs.
    static tensor max(tensor const& lhs, tensor const& rhs) {
        auto ret = lhs;
        for (size_t i = 0; i < tensor_dim_max; ++i)
            ret._sizes[i] = std::max(ret._sizes[i], rhs._sizes[i]);

        return ret;
    }

    /// @brief Returns a tensor containing values minimum from @p lhs and @p rhs.
    static tensor min(tensor const& lhs, tensor const& rhs) {
        auto ret = lhs;
        for (size_t i = 0; i < tensor_dim_max; ++i)
            ret._sizes[i] = std::min(ret._sizes[i], rhs._sizes[i]);

        return ret;
    }

private:
    /// @brief Helper functions for tensor constructor using dim_vec_kinds
    template <typename KindInitT>
    void assign_inits(KindInitT&& init) {
        init.init_tensor_values(*this);
    }

    template <typename KindInitT, typename... KindInitTys>
    void assign_inits(KindInitT&& init, KindInitTys&&... kind_inits) {
        init.init_tensor_values(*this);
        assign_inits(std::forward<KindInitTys>(kind_inits)...);
    }
};

#define TensorValue(val) static_cast<cldnn::tensor::value_type>(val)

template <details::dim_vec_kind Kind>
inline void details::dim_vec_kind_init<Kind>::init_tensor_values(cldnn::tensor& t) {
    for (size_t i = _dimOffset; i < (size_t)(_dimOffset + _dimSize); i++)
        t._sizes[i] = _sizes[i - _dimOffset];
}

/// @brief Adds two @p tensors
inline tensor operator+(const tensor& lhs, const tensor& rhs) { return lhs.add(rhs); }
/// @brief Subtracts two @p tensors
inline tensor operator-(const tensor& lhs, const tensor& rhs) { return lhs.sub(rhs); }
/// @brief Multiplies a @p tensor to a @p scalar
inline tensor operator*(const tensor& lhs, tensor::value_type rhs) { return lhs.mul(rhs); }
/// @brief Divides a @p tensor by a @p scalar
inline tensor operator/(const tensor& lhs, tensor::value_type rhs) { return lhs.div(rhs); }

/// @}
/// @}
}  // namespace cldnn
