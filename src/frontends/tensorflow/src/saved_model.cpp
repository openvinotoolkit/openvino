// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_iterator_proto.hpp"

template <typename T>
bool _internalReadVariables(std::ifstream& vi_stream, const std::basic_string<T>& path) {
	return false;
}

namespace ov {
namespace frontend {
namespace tensorflow {

bool SavedModelIteratorProto::isValidSignature(const ::tensorflow::SignatureDef& signature) {
    for (const auto& it : signature.inputs()) {
        if (it.second.name().empty()
            //			|| !isRefType(it.second.dtype())
            )
            return false;
    }
    for (const auto& it : signature.outputs()) {
        if (it.second.name().empty()
            //			|| !isRefType(it.second.dtype())
            )
            return false;
    }
    return true;
}

bool SavedModelIteratorProto::readVariables(std::ifstream& vi_stream, const std::string& path) {
	return _internalReadVariables(vi_stream, path);
}

bool SavedModelIteratorProto::readVariables(std::ifstream& vi_stream, const std::wstring& path) {
	return _internalReadVariables(vi_stream, path);
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
