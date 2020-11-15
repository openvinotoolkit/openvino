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
    // TODO [oneDNN] : Set dynamic batch technic is changed for oneDNN. currently is not implemented.
    THROW_IE_EXCEPTION << "currently unsupported method for DNNL";
}