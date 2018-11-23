// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <mkldnn_types.h>
#include "mkldnn_primitive.h"
#include "../../thirdparty/mkl-dnn/src/common/primitive_desc.hpp"
#include "../../thirdparty/mkl-dnn/src/common/memory_pd.hpp"
#include "../../thirdparty/mkl-dnn/src/cpu/cpu_concat.hpp"

using namespace MKLDNNPlugin;

MKLDNNPrimitive::MKLDNNPrimitive() {}

MKLDNNPrimitive::operator bool() {
    return prim ? true : false;
}

mkldnn::primitive MKLDNNPrimitive::operator*() {
    return *prim;
}

void MKLDNNPrimitive::reset(mkldnn::primitive* prim) {
    this->prim.reset(prim);
}

MKLDNNPrimitive &MKLDNNPrimitive::operator=(const std::shared_ptr<mkldnn::primitive>& prim) {
    this->prim = prim;
    return *this;
}

void MKLDNNPrimitive::setBatchLimit(int batch, size_t inputNum, size_t outputNum) {
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
}