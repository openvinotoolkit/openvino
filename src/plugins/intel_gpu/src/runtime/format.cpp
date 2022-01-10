// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/format.hpp"

#include <list>
#include <vector>
#include <algorithm>

namespace cldnn {

const format_traits& format::traits(type fmt) {
    static const std::map<type, format_traits> traits {
            // B - number of Batch dimensions
            // F - number of Feature dimensions
            // S - number of Spatial dimensions
            // G - number of Group dimensions
            // Order - dims changing order from rare to often
            // Inner order - dims order for internal storage in _sizes array
            // Block sizes - vector of pairs of dimension number (by inner order) and block size ordered from rare to often
            // Format                  B  F  S  G   Dims order   Order  Inner order  Block sizes
            { yxfb,                  { 1, 1, 2, 0, {2, 3, 1, 0},       "yxfb",   "bfxy?",  {}}},
            { byxf,                  { 1, 1, 2, 0, {0, 2, 3, 1},       "byxf",   "bfxy?",  {}}},
            { bfyx,                  { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {}}},
            { fyxb,                  { 1, 1, 2, 0, {1, 2, 3, 0},       "fyxb",   "bfxy?",  {}}},
            { b_fs_yx_fsv16,         { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy",   {{1, 16}}}},
            { b_fs_yx_fsv32,         { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy",   {{1, 32}}}},
            { b_fs_zyx_fsv32,        { 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{1, 32}}}},
            { bs_xs_xsv8_bsv8,       { 1, 0, 1, 0, {0, 1},             "bx",     "b?x??",  {{2, 8}, {0, 8}}}},
            { bs_xs_xsv8_bsv16,      { 1, 0, 1, 0, {0, 1},             "bx",     "b?x??",  {{2, 8}, {0, 16}}}},
            { bs_x_bsv16,            { 1, 1, 1, 0, {0, 1},             "bx",     "b?x??",  {{0, 16}}}},
            { winograd_2x3_s1_data,  { 1, 1, 2, 0, {0, 2, 3, 1},       "bxyf",   "bfxy?",  {}}},
            { b_fs_yx_fsv4,          { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{1, 4}}}},
            { bfzyx,                 { 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {}}},
            { bfwzyx,                { 1, 1, 4, 0, {0, 1, 2, 3, 4, 5}, "bfwzyx", "bfxyzw", {}}},
            { fs_b_yx_fsv32,         { 1, 1, 2, 0, {1, 0, 2, 3},       "fbyx",   "bfxy?",  {{1, 32}}}},
            { b_fs_yx_32fp,          { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {}}},
            { b_fs_zyx_fsv16,        { 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{1, 16}}}},
            { bs_fs_zyx_bsv16_fsv16, { 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{0, 16 }, {1, 16}}}},
            { bs_fs_yx_bsv16_fsv16,  { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 16 }, {1, 16}}}},
            { bs_fs_yx_bsv4_fsv4,    { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 4 }, {1, 4}}}},
            { bs_fs_yx_bsv8_fsv4,    { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 8 }, {1, 4}}}},
            { bs_fs_yx_bsv4_fsv2,    { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 4 }, {1, 2}}}},
            { bs_fs_zyx_bsv4_fsv4,   { 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{0, 4 }, {1, 4}}}},
            { bs_fs_zyx_bsv4_fsv2,   { 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{0, 4 }, {1, 2}}}},
            { bs_fs_zyx_bsv32_fsv32, { 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{0, 32 }, {1, 32}}}},
            { bs_fs_zyx_bsv32_fsv16, { 1, 1, 3, 0, {0, 1, 2, 3, 4},    "bfzyx",  "bfxyz",  {{0, 32 }, {1, 16}}}},
            { bs_fs_yx_bsv32_fsv32,  { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 32 }, {1, 32}}}},
            { bs_fs_yx_bsv32_fsv16,  { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {{0, 32 }, {1, 16}}}},
            { nv12,                  { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {}}},
            { image_2d_rgba,         { 1, 1, 2, 0, {0, 1, 2, 3},       "bfyx",   "bfxy?",  {}}},

            { oiyx,                                        { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",       {}}},
            { ioyx,                                        { 1, 1, 2, 0, {1, 0, 2, 3},    "ioyx",   "oixy",       {}}},
            { iyxo,                                        { 1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",       {}}},
            { oyxi,                                        { 1, 1, 2, 0, {0, 2, 3, 1},    "oyxi",   "oixy",       {}}},
            { yxio,                                        { 1, 1, 2, 0, {2, 3, 1, 0},    "yxio",   "oixy?",      {}}},
            { oizyx,                                       { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {}}},
            { iozyx,                                       { 1, 1, 3, 0, {1, 0, 2, 3, 4}, "iozyx",  "oixyz",      {}}},
            { os_is_yx_isv16_osv16,                        { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",       {{1, 16}, {0, 16}}}},
            { o_is_yx_isv16,                               { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {{1, 16}}}},
            { os_yxi_osv16,                                { 1, 1, 2, 0, {0, 2, 3, 1},    "oyxi",   "oixy?",      {{0, 16}}}},
            { os_iyx_osv16,                                { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {{0, 16}}}},
            { os_iyx_osv32,                                { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {{0, 32}}}},
            { os_iyx_osv64,                                { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {{0, 64}}}},
            { winograd_2x3_s1_weights,                     { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {}}},
            { winograd_2x3_s1_fused_weights,               { 1, 1, 2, 0, {3, 2, 1, 0},    "xyio",   "oixy?",      {}}},
            { winograd_6x3_s1_fused_weights,               { 1, 1, 2, 0, {3, 2, 1, 0},    "xyio",   "oixy?",      {}}},
            { image_2d_weights_winograd_6x3_s1_fbxyb,      { 1, 1, 2, 0, {3, 2, 1, 0},    "xyio",   "oixy?",      {}}},
            { image_2d_weights_winograd_6x3_s1_xfbyb,      { 1, 1, 2, 0, {3, 2, 1, 0},    "xyio",   "oixy?",      {}}},
            { image_2d_weights_c4_fyx_b,                   { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {}}},
            { image_2d_weights_c1_b_fyx,                   { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {}}},
            { lstm_weights_dio,                            { 1, 1, 2, 0, {0, 1, 3, 2},    "oixy",   "oixy?",      {}}},
            { os_is_yx_isa8_osv8_isv4,                     { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {}}},
            { os_is_yx_isa8_osv16_isv4,                    { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {}}},
            { os_is_yx_isa8_osv8_isv4_swizzled_by_4,       { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {}}},
            { os_is_yx_osa4_isa8_osv8_isv2,                { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {{0, 32}, {1, 16}}}},
            { os_is_yx_osa4_isa8_osv8_isv4,                { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",      {{0, 32}, {1, 32}}}},
            { os_is_zyx_osa4_isa8_osv8_isv2,               { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {{0, 32}, {1, 16}}}},
            { os_is_zyx_osa4_isa8_osv8_isv4,               { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {{0, 32}, {1, 32}}}},
            { os_is_yx_osa2_isa8_osv16_isv2,               { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",       {{0, 32}, {1, 16}}}},
            { os_is_yx_osa2_isa8_osv16_isv4,               { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",       {{0, 32}, {1, 32}}}},
            { os_is_yx_osa2_isa8_osv8_isv2,                { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",       {{0, 16}, {1, 16}}}},
            { os_is_zyx_isa8_osv8_isv4,                    { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {{1, 8}, {0, 8}, {1, 4}}}},
            { os_is_zyx_isa8_osv16_isv4,                   { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {{1, 8}, {0, 16}, {1, 4}}}},
            { os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,  { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy?",      {{0, 32}, {1, 32}}}},
            { os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4, { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {{0, 32}, {1, 32}}}},
            { is_os_yx_isa2_osa8_isv8_osv2,                { 1, 1, 2, 0, {1, 0, 2, 3}, "ioyx",   "ioxy?",      {{1, 16}, {0, 16}}}},
            { is_o_yx_isv32,                               { 1, 1, 2, 0, {1, 0, 2, 3},    "oyxi",   "oixy?",      {{1, 32}}}},
            { is_o32_yx_isv32_swizzled_by_4,               { 1, 1, 2, 0, {0, 1, 2, 3},    "oyxi",   "oixy?",      {}}},
            { os_is_y_x8_osv8_isv4,                        { 1, 1, 2, 0, {0, 1, 2, 3},    "oyxi",   "oixy?",      {}}},
            { os_is_y_x8_osv8_isv4_swizzled_by_4,          { 1, 1, 2, 0, {0, 1, 2, 3},    "oyxi",   "oixy?",      {}}},
            { os_is_yx_osv16_isv4,                         { 1, 1, 2, 0, {0, 1, 2, 3},    "oixy",   "oixy?",      {{0, 16}, {1, 4}}}},
            { os_is_yx_osv8_isv4,                          { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",       {{1, 4}, {0, 8}}}},
            { os_is_yx_osv8_isv2,                          { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",       {{1, 2}, {0, 8}}}},
            { os_is_zyx_osv16_isv16,                       { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {{0, 16}, {1, 16}}}},
            { os_is_yx_osv32_isv4_swizzled_by_2,           { 1, 1, 2, 0, {0, 1, 2, 3},    "oixy",   "oixy?",      {{0, 32}, {1, 4}}}},
            { os_is_yx_osv32_isv4,                         { 1, 1, 2, 0, {0, 1, 2, 3},    "oixy",   "oixy?",      {{0, 32}, {1, 4}}}},
            { os_is_zyx_osv32_isv4,                        { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {{0, 32}, {1, 4}}}},
            { os_is_yx_osv32_isv32p,                       { 1, 1, 1, 0, {0, 1, 2, 3},    "oixy",   "oixy?",      {}}},
            { os_is_zyx_isv16_osv16,                       { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {{0, 16}, {1, 16}}}},
            { is_os_zyx_isv16_osv16,                       { 1, 1, 3, 0, {1, 0, 2, 3, 4}, "iozyx",  "oixyz",      {{1, 16}, {0, 16}}}},
            { is_os_yx_isv16_osv16,                        { 1, 1, 2, 0, {1, 0, 2, 3, 4}, "ioyx",   "oixy",      {{1, 16}, {0, 16}}}},
            { os_is_osv32_isv32_swizzled_by_4,             { 1, 1, 0, 0, {0, 1, 2, 3},    "oixy",   "oixy?",      {{0, 32}, {1, 32}}}},
            { os_is_zyx_isv8_osv16_isv2,                   { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {{1, 8}, {0, 16}, {1, 2}}}},
            { os_zyxi_osv16,                               { 1, 1, 3, 0, {0, 2, 3, 4, 1}, "ozyxi",  "oixyz",      {{0, 16}}}},
            { os_is_yx_isv8_osv16_isv2,                    { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",       {{1, 8}, {0, 16}, {1, 2}}}},
            { os_is_yx_osv16_isv16,                        { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",       {{1, 16}, {0, 16}}}},
            { os_is_zyx_osv32_isv16,                       { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {{0, 32}, {1, 16}}}},
            { os_is_zyx_osv64_isv16,                       { 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx",  "oixyz",      {{0, 64}, {1, 16}}}},
            { os_iyx_osv32__ai32,                          { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",       {{0, 32}}}},
            { i_yxs_os_yxsv2_osv16,                        { 1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",       {{0, 16}}}},
            { iy_xs_os_xsv2_osv8__ao32,                    { 1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",       {{2, 2}, {0, 8}}}},
            { iy_xs_os_xsv2_osv16__ao32,                   { 1, 1, 2, 0, {1, 2, 3, 0},    "iyxo",   "oixy",       {{2, 2}, {0, 16}}}},
            { os_i_yxs_osv4_yxsv4,                         { 1, 1, 2, 0, {0, 1, 2, 3},    "oiyx",   "oixy",       {{0, 4}}}},
            { os_i_osv16__ai8,                             { 1, 1, 0, 0, {0, 1},          "oi",     "oi??",       {{1, 8}, {0, 16}}}},
            { os_i_osv8__ai8,                              { 1, 1, 0, 0, {0, 1},          "oi",     "oi??",       {{1, 8}, {0, 8}}}},

            { goiyx,                                       { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {}}},
            { gioyx,                                       { 1, 1, 2, 1, {0, 2, 1, 3, 4},    "gioyx",  "oixy??g",  {}}},
            { goizyx,                                      { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g",  {}}},
            { giozyx,                                      { 1, 1, 3, 1, {0, 2, 1, 3, 4, 5}, "giozyx", "oixyz?g",  {}}},
            { g_os_iyx_osv16,                              { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{0, 16}}}},
            { g_os_iyx_osv32,                              { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{0, 32}}}},
            { gs_oiyx_gsv16,                               { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{6, 16}}}},
            { gs_oizyx_gsv16,                              { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g",  {{6, 16}}}},
            { gs_oiyx_gsv32,                               { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{6, 32}}}},
            { gyxio,                                       { 1, 1, 2, 1, {0, 3, 4, 2, 1},    "gyxio",  "oixy??g",  {}}},
            { g_is_os_zyx_isv16_osv16,                     { 1, 1, 3, 1, {0, 2, 1, 3, 4, 5}, "giozyx", "oixyz?g",  {{1, 16}, {0, 16}}}},
            { g_is_os_yx_isv16_osv16,                      { 1, 1, 2, 1, {0, 2, 1, 3, 4},    "gioyx",  "oixy??g",  {{1, 16}, {0, 16}}}},
            { g_os_is_zyx_isv8_osv16_isv2,                 { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g",  {{1, 8}, {0, 16}, {1, 2}}}},
            { g_os_is_yx_isv8_osv16_isv2,                  { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{1, 8}, {0, 16}, {1, 2}}}},
            { g_os_is_zyx_isv16_osv16,                     { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g",  {{0, 16}, {1, 16}}}},
            { g_os_is_yx_osv16_isv4,                       { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goixy",  "oixy??g",  {{0, 16}, {1, 4}}}},
            { g_os_is_zyx_osv16_isv16,                     { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g",  {{0, 16}, {1, 16}}}},
            { g_os_zyx_is_osv16_isv4,                      { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g",  {{0, 16}, {1, 4}}}},
            { g_os_zyx_is_osv16_isv16,                     { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g",  {{0, 16}, {1, 16}}}},
            { g_os_zyx_is_osv16_isv32,                     { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g",  {{0, 16}, {1, 32}}}},
            { g_os_zyx_is_osv32_isv4,                      { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g",  {{0, 32}, {1, 4}}}},
            { g_os_zyx_is_osv32_isv16,                     { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g",  {{0, 32}, {1, 16}}}},
            { g_os_zyx_is_osv32_isv32,                     { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "gozyxi", "oixyz?g",  {{0, 32}, {1, 32}}}},
            { g_os_is_yx_osa4_isa8_osv8_isv4,              { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{0, 32}, {1, 32}}}},
            { g_os_is_zyx_osa4_isa8_osv8_isv4,             { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g",  {{0, 32}, {1, 32}}}},
            { g_os_is_yx_osa4_isa8_osv8_isv2,              { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{0, 32}, {1, 16}}}},
            { g_os_is_zyx_osa4_isa8_osv8_isv2,             { 1, 1, 3, 1, {0, 1, 2, 3, 4, 5}, "goizyx", "oixyz?g",  {{0, 32}, {1, 16}}}},
            { g_os_is_yx_osa2_isa8_osv16_isv4,             { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{0, 32}, {1, 32}}}},
            { g_os_is_yx_osa2_isa8_osv16_isv2,             { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{0, 32}, {1, 16}}}},
            { gs_oi_yxs_gsv4_yxsv4,                        { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{6, 4}}}},
            { gs_oi_yxs_gsv16_yxsv4,                       { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{6, 16}}}},
            { gs_oi_yxs_gsv32_yxsv4,                       { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{6, 32}}}},
            { g_os_is_yx_isv16_osv16,                      { 1, 1, 2, 1, {0, 1, 2, 3, 4},    "goiyx",  "oixy??g",  {{1, 16}, {0, 16}}}},
            { gi_yxs_os_yxsv2_osv16,                       { 1, 1, 2, 1, {0, 2, 3, 4, 1},    "giyxo",  "oixy??g",  {{0, 16}}}},
            { giy_xs_os_xsv2_osv8__ao32,                   { 1, 1, 2, 1, {0, 2, 3, 4, 1},    "giyxo",  "oixy??g",  {{2, 2}, {0, 8}}}},
            { giy_xs_os_xsv2_osv16__ao32,                  { 1, 1, 2, 1, {0, 2, 3, 4, 1},    "giyxo",  "oixy??g",  {{2, 2}, {0, 16}}}},
    };
    if (traits.find(fmt) == traits.end()) {
        throw std::runtime_error("[clDNN] Format description is missing in fmt traits");
    }
    return traits.at(fmt);
}

std::string format::to_string() const {
    switch (value) {
        case format::yxfb:
            return "yxfb";
        case format::byxf:
            return "byxf";
        case format::bfyx:
            return "bfyx";
        case format::fyxb:
            return "fyxb";
        case format::b_fs_yx_fsv16:
            return "b_fs_yx_fsv16";
        case format::b_fs_yx_fsv32:
            return "b_fs_yx_fsv32";
        case format::b_fs_zyx_fsv32:
            return "b_fs_zyx_fsv32";
        case format::bs_xs_xsv8_bsv8:
            return "bs_xs_xsv8_bsv8";
        case format::bs_xs_xsv8_bsv16:
            return "bs_xs_xsv8_bsv16";
        case format::bs_x_bsv16:
            return "bs_x_bsv16";
        case format::winograd_2x3_s1_data:
            return "winograd_2x3_s1_data";
        case format::b_fs_yx_fsv4:
            return "b_fs_yx_fsv4";
        case format::b_fs_yx_32fp:
            return "b_fs_yx_32fp";
        case format::bfzyx:
            return "bfzyx";
        case format::bfwzyx:
            return "bfwzyx";
        case format::fs_b_yx_fsv32:
            return "fs_b_yx_fsv32";
        case format::bs_fs_yx_bsv16_fsv16:
            return "bs_fs_yx_bsv16_fsv16";
        case format::bs_fs_yx_bsv32_fsv16:
            return "bs_fs_yx_bsv32_fsv16";
        case format::bs_fs_yx_bsv4_fsv2:
            return "bs_fs_yx_bsv4_fsv2";
        case format::bs_fs_zyx_bsv4_fsv2:
            return "bs_fs_zyx_bsv4_fsv2";
        case format::bs_fs_zyx_bsv4_fsv4:
            return "bs_fs_zyx_bsv4_fsv4";
        case format::bs_fs_yx_bsv4_fsv4:
            return "bs_fs_yx_bsv4_fsv4";
        case format::bs_fs_yx_bsv8_fsv4:
            return "bs_fs_yx_bsv8_fsv4";
        case format::bs_fs_zyx_bsv32_fsv16:
            return "bs_fs_zyx_bsv32_fsv16";
        case format::bs_fs_zyx_bsv32_fsv32:
            return "bs_fs_zyx_bsv32_fsv32";
        case format::bs_fs_yx_bsv32_fsv32:
            return "bs_fs_yx_bsv32_fsv32";
        case format::b_fs_zyx_fsv16:
            return "b_fs_zyx_fsv16";
        case format::bs_fs_zyx_bsv16_fsv16:
            return "bs_fs_zyx_bsv16_fsv16";
        case format::image_2d_rgba:
            return "image_2d_rgba";
        case format::nv12:
            return "nv12";

        case format::oiyx:
            return "oiyx";
        case format::ioyx:
            return "ioyx";
        case format::yxio:
            return "yxio";
        case format::iyxo:
            return "iyxo";
        case format::oyxi:
            return "oyxi";
        case format::oizyx:
            return "oizyx";
        case format::iozyx:
            return "iozyx";
        case format::winograd_2x3_s1_weights:
            return "winograd_2x3_s1_weights";
        case format::winograd_2x3_s1_fused_weights:
            return "winograd_2x3_s1_fused_weights";
        case format::winograd_6x3_s1_fused_weights:
            return "winograd_6x3_s1_fused_weights";
        case format::image_2d_weights_c4_fyx_b:
            return "image_2d_weights_c4_fyx_b";
        case format::image_2d_weights_c1_b_fyx:
            return "image_2d_weights_c1_b_fyx";
        case format::image_2d_weights_winograd_6x3_s1_fbxyb:
            return "image_2d_weights_winograd_6x3_s1_fbxyb";
        case format::image_2d_weights_winograd_6x3_s1_xfbyb:
            return "image_2d_weights_winograd_6x3_s1_xfbyb";
        case format::os_yxi_osv16:
            return "os_yxi_osv16";
        case format::o_is_yx_isv16:
            return "o_is_yx_isv16";
        case format::os_iyx_osv16:
            return "os_iyx_osv16";
        case format::os_is_yx_osv16_isv16:
            return "os_is_yx_osv16_isv16";
        case format::os_is_zyx_osv32_isv16:
            return "os_is_zyx_osv32_isv16";
        case format::os_is_zyx_osv64_isv16:
            return "os_is_zyx_osv64_isv16";
        case format::os_iyx_osv32:
            return "os_iyx_osv32";
        case format::os_iyx_osv64:
            return "os_iyx_osv64";
        case format::is_o_yx_isv32:
            return "is_o_yx_isv32";
        case format::os_is_yx_isv16_osv16:
            return "os_is_yx_isv16_osv16";
        case format::os_is_yx_isa8_osv8_isv4:
            return "os_is_yx_isa8_osv8_isv4";
        case format::os_is_yx_isa8_osv16_isv4:
            return "os_is_yx_isa8_osv16_isv4";
        case format::os_is_zyx_isa8_osv8_isv4:
            return "os_is_zyx_isa8_osv8_isv4";
        case format::os_is_yx_isv8_osv16_isv2:
            return "os_is_yx_isv8_osv16_isv2";
        case format::os_is_y_x8_osv8_isv4_swizzled_by_4:
            return "os_is_y_x8_osv8_isv4_swizzled_by_4";
        case format::os_is_yx_osa2_isa8_osv16_isv4:
            return "os_is_yx_osa2_isa8_osv16_isv4";
        case format::os_is_yx_osa2_isa8_osv16_isv2:
            return "os_is_yx_osa2_isa8_osv16_isv2";
        case format::os_is_yx_osa2_isa8_osv8_isv2:
            return "os_is_yx_osa2_isa8_osv8_isv2";
        case format::os_is_zyx_isa8_osv16_isv4:
            return "os_is_zyx_isa8_osv16_isv4";
        case format::os_is_yx_osa4_isa8_osv8_isv2:
            return "os_is_yx_osa4_isa8_osv8_isv2";
        case format::os_is_zyx_osa4_isa8_osv8_isv2:
            return "os_is_zyx_osa4_isa8_osv8_isv2";
        case format::os_is_zyx_osa4_isa8_osv8_isv4:
            return "os_is_zyx_osa4_isa8_osv8_isv4";
        case format::g_os_is_yx_osa4_isa8_osv8_isv2:
            return "g_os_is_yx_osa4_isa8_osv8_isv2";
        case format::g_os_is_yx_osa4_isa8_osv8_isv4:
            return "g_os_is_yx_osa4_isa8_osv8_isv4";
        case format::g_os_is_zyx_osa4_isa8_osv8_isv4:
            return "g_os_is_zyx_osa4_isa8_osv8_isv4";
        case format::g_os_is_zyx_osa4_isa8_osv8_isv2:
            return "g_os_is_zyx_osa4_isa8_osv8_isv2";
        case format::os_is_yx_osa4_isa8_osv8_isv4:
            return "os_is_yx_osa4_isa8_osv8_isv4";
        case format::is_os_yx_isa2_osa8_isv8_osv2:
            return "is_os_yx_isa2_osa8_isv8_osv2";
        case format::os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4:
            return "os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4";
        case format::os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4:
            return "os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4";
        case format::os_is_yx_isa8_osv8_isv4_swizzled_by_4:
            return "os_is_yx_isa8_osv8_isv4_swizzled_by_4";
        case format::is_o32_yx_isv32_swizzled_by_4:
            return "is_o32_yx_isv32_swizzled_by_4";
        case format::os_is_yx_osv8_isv2:
            return "os_is_yx_osv8_isv2";
        case format::os_is_yx_osv8_isv4:
            return "os_is_yx_osv8_isv4";
        case format::os_is_yx_osv16_isv4:
            return "os_is_yx_osv16_isv4";
        case format::os_is_yx_osv32_isv4_swizzled_by_2:
            return "os_is_yx_osv32_isv4_swizzled_by_2";
        case format::os_is_yx_osv32_isv4:
            return "os_is_yx_osv32_isv4";
        case format::os_is_zyx_osv32_isv4:
            return "os_is_zyx_osv32_isv4";
        case format::os_is_y_x8_osv8_isv4:
            return "os_is_y_x8_osv8_isv4";
        case format::os_is_yx_osv32_isv32p:
            return "os_is_yx_osv32_isv32p";
        case format::os_is_zyx_isv16_osv16:
            return "os_is_zyx_isv16_osv16";
        case format::is_os_zyx_isv16_osv16:
            return "is_os_zyx_isv16_osv16";
        case format::is_os_yx_isv16_osv16:
            return "is_os_yx_isv16_osv16";
        case format::os_is_osv32_isv32_swizzled_by_4:
            return "os_is_osv32_isv32_swizzled_by_4";
        case format::os_is_zyx_osv16_isv16:
            return "os_is_zyx_osv16_isv16";
        case format::os_is_zyx_isv8_osv16_isv2:
            return "os_is_zyx_isv8_osv16_isv2";
        case format::os_zyxi_osv16:
            return "os_zyxi_osv16";
        case format::os_i_osv8__ai8:
            return "os_i_osv8__ai8";
        case format::os_i_osv16__ai8:
            return "os_i_osv16__ai8";
        case format::os_iyx_osv32__ai32:
            return "os_iyx_osv32__ai32";
        case format::iy_xs_os_xsv2_osv8__ao32:
            return "iy_xs_os_xsv2_osv8__ao32";
        case format::iy_xs_os_xsv2_osv16__ao32:
            return "iy_xs_os_xsv2_osv16__ao32";
        case format::i_yxs_os_yxsv2_osv16:
            return "i_yxs_os_yxsv2_osv16";
        case format::os_i_yxs_osv4_yxsv4:
            return "os_i_yxs_osv4_yxsv4";
        case format::lstm_weights_dio:
            return "lstm_weights_dio";

        case format::goiyx:
            return "goiyx";
        case format::gyxio:
            return "gyxio";
        case format::goizyx:
            return "goizyx";
        case format::gioyx:
            return "gioyx";
        case format::giozyx:
            return "giozyx";
        case format::g_os_iyx_osv16:
            return "g_os_iyx_osv16";
        case format::g_os_iyx_osv32:
            return "g_os_iyx_osv32";
        case format::gs_oiyx_gsv16:
            return "gs_oiyx_gsv16";
        case format::gs_oiyx_gsv32:
            return "gs_oiyx_gsv32";
        case format::g_is_os_zyx_isv16_osv16:
            return "g_is_os_zyx_isv16_osv16";
        case format::g_is_os_yx_isv16_osv16:
            return "g_is_os_yx_isv16_osv16";
        case format::g_os_is_zyx_isv8_osv16_isv2:
            return "g_os_is_zyx_isv8_osv16_isv2";
        case format::g_os_is_yx_isv8_osv16_isv2:
            return "g_os_is_yx_isv8_osv16_isv2";
        case format::g_os_is_zyx_isv16_osv16:
            return "g_os_is_zyx_isv16_osv16";
        case format::g_os_is_yx_osv16_isv4:
            return "g_os_is_yx_osv16_isv4";
        case format::g_os_is_zyx_osv16_isv16:
            return "g_os_is_zyx_osv16_isv16";
        case format::g_os_zyx_is_osv16_isv4:
            return "g_os_zyx_is_osv16_isv4";
        case format::g_os_zyx_is_osv16_isv16:
            return "g_os_zyx_is_osv16_isv16";
        case format::g_os_zyx_is_osv16_isv32:
            return "g_os_zyx_is_osv16_isv32";
        case format::g_os_zyx_is_osv32_isv4:
            return "g_os_zyx_is_osv32_isv4";
        case format::g_os_zyx_is_osv32_isv16:
            return "g_os_zyx_is_osv32_isv16";
        case format::g_os_zyx_is_osv32_isv32:
            return "g_os_zyx_is_osv32_isv32";
        case format::gs_oi_yxs_gsv32_yxsv4:
            return "gs_oi_yxs_gsv32_yxsv4";
        case format::g_os_is_yx_osa2_isa8_osv16_isv2:
            return "g_os_is_yx_osa2_isa8_osv16_isv2";
        case format::g_os_is_yx_osa2_isa8_osv16_isv4:
            return "g_os_is_yx_osa2_isa8_osv16_isv4";
        case format::giy_xs_os_xsv2_osv8__ao32:
            return "giy_xs_os_xsv2_osv8__ao32";
        case format::giy_xs_os_xsv2_osv16__ao32:
            return "giy_xs_os_xsv2_osv16__ao32";
        case format::gi_yxs_os_yxsv2_osv16:
            return "gi_yxs_os_yxsv2_osv16";
        case format::gs_oi_yxs_gsv16_yxsv4:
            return "gs_oi_yxs_gsv16_yxsv4";
        case format::gs_oi_yxs_gsv4_yxsv4:
            return "gs_oi_yxs_gsv4_yxsv4";
        case format::g_os_is_yx_isv16_osv16:
            return "g_os_is_yx_isv16_osv16";
        case format::gs_oizyx_gsv16:
            return "gs_oizyx_gsv16";
        default:
            throw std::runtime_error("[GPU] format::to_string: unknown layout: " + std::to_string(value));
    }
}

}  // namespace cldnn
