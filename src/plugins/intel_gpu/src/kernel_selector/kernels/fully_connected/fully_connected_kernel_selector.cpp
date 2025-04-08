// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_selector.h"
#include "fully_connected_kernel_bfyx_ref.h"
#include "fully_connected_kernel_bf_io_gemm.h"
#include "fully_connected_kernel_bs_f_bsv16_b1.h"
#include "fully_connected_kernel_bs_f_bsv16_af8.h"
#include "fully_connected_kernel_bs_f_bsv8_af8.h"
#include "fully_connected_kernel_yxfb_ref.h"
#include "fully_connected_kernel_fb_oi_ref.h"
#include "fully_connected_kernel_fb_io_ref.h"
#include "fully_connected_kernel_bf_io_ref.h"
#include "fully_connected_kernel_fb_oi_b8_ref.h"
#include "fully_connected_kernel_fb_io_b8_f8.h"
#include "fully_connected_kernel_fb_io_block.h"
#include "fully_connected_kernel_bf_io_input_spatial.h"
#include "fully_connected_kernel_mmad.h"
#include "fully_connected_kernel_imad.h"
#include "fully_connected_kernel_fs_byx_fsv32.h"
#include "fully_connected_kernel_bf_tiled.h"

namespace kernel_selector {

fully_connected_kernel_selector::fully_connected_kernel_selector() {
    Attach<FullyConnected_bfyx_Ref>();
    Attach<FullyConnected_bf_io_GEMM>();
    Attach<FullyConnected_bs_f_bsv16_b1>();
    Attach<FullyConnected_bs_f_bsv16_af8>();
    Attach<FullyConnected_bs_f_bsv8_af8>();
    Attach<FullyConnected_yxfb_ref>();
    Attach<FullyConnected_fb_oi_ref>();
    Attach<FullyConnected_fb_io_ref>();
    Attach<FullyConnected_bf_io_ref>();
    Attach<FullyConnected_fb_oi_b8_ref>();
    Attach<FullyConnected_fb_io_block>();
    Attach<FullyConnected_fb_io_b8_f8>();
    Attach<FullyConnected_bf_io_input_spatial>();
    Attach<FullyConnectedKernelMMAD>();
    Attach<FullyConnectedKernelIMAD>();
    Attach<FullyConnected_fs_byx_fsv32>();
    Attach<FullyConnected_bf_tiled>();
}

KernelsData fully_connected_kernel_selector::GetBestKernels(const Params& params) const {
    return GetAutoTuneBestKernel(params, KernelType::FULLY_CONNECTED);
}
}  // namespace kernel_selector
