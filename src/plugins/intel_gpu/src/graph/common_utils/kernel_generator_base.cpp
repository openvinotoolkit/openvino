// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_generator_base.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include "micro_utils.hpp"
#endif

#include <cctype>

namespace ov::intel_gpu {

void KernelData::save(cldnn::BinaryOutputBuffer& ob) const {
    ob(params.workGroups.global, params.workGroups.local);
    ob << params.arguments.size();
    for (const auto& arg : params.arguments) {
        ob << make_data(&arg.t, sizeof(cldnn::argument_desc::Types)) << arg.index;
    }
    ob << params.scalars.size();
    for (const auto& scalar : params.scalars) {
        ob << make_data(&scalar.t, sizeof(cldnn::scalar_desc::Types)) << make_data(&scalar.v, sizeof(cldnn::scalar_desc::ValueT));
    }
    ob << params.layerID;
#ifdef ENABLE_ONEDNN_FOR_GPU
    ob << micro_kernels.size();
    for (const auto& microkernel : micro_kernels) {
        microkernel->save(ob);
    }
#endif
}

void KernelData::load(cldnn::BinaryInputBuffer& ib) {
    ib(params.workGroups.global, params.workGroups.local);

    typename cldnn::arguments_desc::size_type arguments_desc_size = 0UL;
    ib >> arguments_desc_size;
    params.arguments.resize(arguments_desc_size);
    for (auto& arg : params.arguments) {
        ib >> make_data(&arg.t, sizeof(cldnn::argument_desc::Types)) >> arg.index;
    }

    typename cldnn::scalars_desc::size_type scalars_desc_size = 0UL;
    ib >> scalars_desc_size;
    params.scalars.resize(scalars_desc_size);
    for (auto& scalar : params.scalars) {
        ib >> make_data(&scalar.t, sizeof(cldnn::scalar_desc::Types)) >> make_data(&scalar.v, sizeof(cldnn::scalar_desc::ValueT));
    }

    ib >> params.layerID;

#ifdef ENABLE_ONEDNN_FOR_GPU
    size_t n_microkernels = 0;
    ib >> n_microkernels;
    micro_kernels.clear();
    for (size_t i = 0; i < n_microkernels; i++) {
        auto microkernel = std::make_shared<micro::MicroKernelPackage>();
        microkernel->load(ib);
        micro_kernels.push_back(microkernel);
    }
#endif
}

}  // namespace ov::intel_gpu
