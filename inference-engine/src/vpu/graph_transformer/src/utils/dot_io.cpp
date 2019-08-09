// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/dot_io.hpp>

#include <string>
#include <iostream>
#include <algorithm>
#include <vector>

#include <precision_utils.h>

#include <vpu/utils/any.hpp>
#include <vpu/utils/attributes_map.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

DotLabel::DotLabel(const std::string& caption, DotSerializer& out) : _out(out) {
    _ostr << "label=\"" << caption << "\\l";
}

DotLabel::DotLabel(DotLabel& other) : _out(other._out), _parent(&other), _ident(other._ident) {
    ++_ident;
    _ostr << "[\\l";
}

DotLabel::~DotLabel() {
    if (_parent == nullptr) {
        _ostr << "\"";

        try {
            _out.append("%s", _ostr.str());
        }
        catch (...) {
            std::cerr << "ERROR ~DotLabel(): can not append symbols\n";
        }

    } else {
        --_ident;

        for (size_t i = 0; i < _ident; ++i)
            _ostr << "    ";

        _ostr << "]";

        _parent->_ostr << _ostr.str();
    }
}

void DotLabel::addIdent() {
    for (size_t i = 0; i < _ident; ++i)
        _ostr << "    ";
}

void printTo(DotLabel& lbl, const Any& any) {
    any.printImpl(lbl);
}

void printTo(DotLabel& lbl, const AttributesMap& attrs) {
    attrs.printImpl(lbl);
}

void printTo(DotLabel& lbl, const ie::DataPtr& ieData) {
    IE_ASSERT(ieData != nullptr);

    DotLabel subLbl(lbl);
    subLbl.appendPair("name", ieData->getName());
    subLbl.appendPair("precision", ieData->getTensorDesc().getPrecision().name());
    subLbl.appendPair("dims", ieData->getTensorDesc().getDims());
    subLbl.appendPair("layout", ieData->getTensorDesc().getLayout());
}

void printTo(DotLabel& lbl, const ie::Blob::Ptr& ieBlob) {
    IE_ASSERT(ieBlob != nullptr);

    DotLabel subLbl(lbl);
    subLbl.appendPair("precision", ieBlob->getTensorDesc().getPrecision().name());
    subLbl.appendPair("dims", ieBlob->getTensorDesc().getDims());
    subLbl.appendPair("layout", ieBlob->getTensorDesc().getLayout());

    if (ieBlob->getTensorDesc().getPrecision() == ie::Precision::FP32) {
        auto contentPtr = ieBlob->cbuffer().as<const uint8_t*>();
        auto count = ieBlob->size();

        SmallVector<uint8_t, 8> temp(
            contentPtr,
            contentPtr + std::min<int>(count, 8));

        subLbl.appendPair("content", temp);
    } else if (ieBlob->getTensorDesc().getPrecision() == ie::Precision::FP16) {
        auto contentPtr = ieBlob->cbuffer().as<const fp16_t*>();
        auto count = ieBlob->size();

        SmallVector<float, 8> temp(std::min<int>(count, 8));
        ie::PrecisionUtils::f16tof32Arrays(temp.data(), contentPtr, temp.size());

        lbl.appendPair("content", temp);
    }
}

void printTo(DotLabel& lbl, const ie::CNNLayerPtr& ieLayer) {
    IE_ASSERT(ieLayer != nullptr);

    DotLabel subLbl(lbl);
    subLbl.appendPair("name", ieLayer->name);
    subLbl.appendPair("type", ieLayer->type);
    subLbl.appendPair("precision", ieLayer->precision.name());
}

}  // namespace vpu
