// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mkldnn_types.h>
#include "mkldnn_primitive.h"

using namespace MKLDNNPlugin;

MKLDNNPrimitive::MKLDNNPrimitive() {}

MKLDNNPrimitive::operator bool() {
    return prim ? true : false;
}

mkldnn::primitive MKLDNNPrimitive::operator*() {
    return *prim;
}

void MKLDNNPrimitive::reset(mkldnn::primitive* primitive) {
    prim.reset(primitive);
}

MKLDNNPrimitive &MKLDNNPrimitive::operator=(const std::shared_ptr<mkldnn::primitive>& primitive) {
    prim = primitive;
    return *this;
}

void MKLDNNPrimitive::setBatchLimit(int batch, size_t inputNum, size_t outputNum) {
#ifdef USE_DNNL
    THROW_IE_EXCEPTION << "currently unsupported method for DNNL";
#else
    bool success = true;
    auto * primDesc = prim->get_primitive_desc();
    auto * concatPrimDesc = dynamic_cast<const mkldnn::impl::cpu::cpu_concat_pd_t *>(primDesc);
    for (int i = 0; success && i < primDesc->n_inputs() && i < inputNum; i++) {
        // Depthwise layers contains weights as input
        if (primDesc->input_pd()->desc()->ndims != primDesc->input_pd(i)->desc()->ndims)
            break;
        auto * memDesc = const_cast<mkldnn_memory_desc_t *>(primDesc->input_pd(i)->desc());
        if (originInputBatches.size() <= i)
            originInputBatches.push_back(memDesc->dims[0]);

        if (batch > originInputBatches[i])
            success = false;
        memDesc->dims[0] = batch;
        memDesc->layout_desc.blocking.padding_dims[0] = batch;
        if (concatPrimDesc != nullptr) {
            memDesc = const_cast<mkldnn_memory_desc_t *>(concatPrimDesc->src_image_pd(i)->desc());
            memDesc->dims[0] = batch;
            memDesc->layout_desc.blocking.padding_dims[0] = batch;
        }
    }
    for (int i = 0; success && i < primDesc->n_outputs() && i < outputNum; i++) {
        if (primDesc->output_pd()->desc()->ndims != primDesc->output_pd(i)->desc()->ndims)
            break;
        auto * memDesc = const_cast<mkldnn_memory_desc_t *>(primDesc->output_pd(i)->desc());
        if (i < inputNum && memDesc == primDesc->input_pd(i)->desc())
            continue;
        if (originOutputBatches.size() <= i)
            originOutputBatches.push_back(memDesc->dims[0]);

        if (batch > originOutputBatches[i])
            success = false;
        memDesc->dims[0] = batch;
        memDesc->layout_desc.blocking.padding_dims[0] = batch;
    }

    if (success)
        return;

    for (int i = 0; i < primDesc->n_inputs() && i < originInputBatches.size(); i++) {
        auto * memDesc = const_cast<mkldnn_memory_desc_t *>(primDesc->input_pd(i)->desc());
        memDesc->dims[0] = originInputBatches[i];
        memDesc->layout_desc.blocking.padding_dims[0] = originInputBatches[i];
    }
    for (int i = 0; i < primDesc->n_outputs() && i < originOutputBatches.size(); i++) {
        auto * memDesc = const_cast<mkldnn_memory_desc_t *>(primDesc->output_pd(i)->desc());
        memDesc->dims[0] = originOutputBatches[i];
        memDesc->layout_desc.blocking.padding_dims[0] = originOutputBatches[i];
    }

    THROW_IE_EXCEPTION << "Dynamic batch cannot be changed!";
#endif
}