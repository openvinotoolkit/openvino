// Copyright (c) 2016-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "fused_conv_eltwise_kernel_selector.h"
#include "fused_conv_eltwise_kernel_gemm.h"
#include "fused_conv_eltwise_kernel_bfyx_1x1_opt.h"
#include "fused_conv_eltwise_kernel_bfyx_os_iyx_osv16.h"
#include "fused_conv_eltwise_kernel_mmad_32x32sg_128x128wg_slm_int8.h"
#include "fused_conv_eltwise_kernel_mmad_32x32sg_224x128wg_slm_int8.h"
#include "fused_conv_eltwise_kernel_yxfb_yxio_b16.h"
#include "fused_conv_eltwise_kernel_imad.h"
#include "fused_conv_eltwise_kernel_af32_imad_1x1.h"

namespace kernel_selector {
fused_conv_eltwise_kernel_selector::fused_conv_eltwise_kernel_selector() {
    //        Attach<fused_conv_eltwise_kernel_gemm>();
    Attach<fused_conv_eltwise_kernel_yxfb_yxio_b16>();
    Attach<fused_conv_eltwise_kernel_bfyx_1x1_opt>();
    Attach<fused_conv_eltwise_kernel_bfyx_os_iyx_osv16>();
    Attach<fused_conv_eltwise_kernel_mmad_32x32sg_128x128wg_slm_int8>();
    Attach<fused_conv_eltwise_kernel_mmad_32x32sg_224x128wg_slm_int8>();
    Attach<fused_conv_eltwise_kernel_imad>();
    Attach<fused_conv_eltwise_kernel_af32_imad_1x1>();
}

KernelsData fused_conv_eltwise_kernel_selector::GetBestKernels(const Params& params,
                                                               const optional_params& options) const {
    return GetAutoTuneBestKernel(params, options, KernelType::FUSED_CONV_ELTWISE);
}
}  // namespace kernel_selector
