// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/any.hpp>
#include <vpu/utils/attributes_map.hpp>

#include <precision_utils.h>

#include <string>

namespace vpu {

DotLabel::DotLabel(const std::string& caption, DotSerializer& out) : _out(out) {
    _ostr << "label=\"" << caption << "\\l";
}

DotLabel::DotLabel(DotLabel& other) : _out(other._out), _parent(&other), _ident(other._ident) {
    ++_ident;
    _ostr << "[\\l";
}

DotLabel::~DotLabel() {
    try {
        if (_parent == nullptr) {
            _ostr << '"';

            _out.append("%s", _ostr.str());
        } else {
            assert(_ident > 0);
            --_ident;

            addIdent();
            _ostr << "]";

            _parent->_ostr << _ostr.str();
        }
    }
    catch (...) {
        std::cerr << "ERROR in ~DotLabel()" << std::endl;
    }
}

void DotLabel::addIdent() {
    for (size_t i = 0; i < _ident; ++i) {
        _ostr << "    ";
    }
}

}  // namespace vpu
