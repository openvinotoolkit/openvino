// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_common/parser.hpp"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <onnx/onnx_pb.h>

#include "openvino/core/except.hpp"
#include "openvino/util/file_util.hpp"

using namespace ::ONNX_NAMESPACE;

namespace ov {
namespace frontend {
namespace onnx {
namespace common {
ModelProto parse_from_file(const std::string& file_path) {
    std::ifstream file_stream{file_path.c_str(), std::ios::in | std::ios::binary};

    if (!file_stream.is_open()) {
        OPENVINO_THROW("Could not open the file: \"" + file_path, '"');
    };

    auto model_proto = parse_from_istream(file_stream);
    file_stream.close();
    return model_proto;
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
ModelProto parse_from_file(const std::wstring& file_path) {
    std::ifstream file_stream{file_path.c_str(), std::ios::in | std::ios::binary};

    if (!file_stream.is_open()) {
        OPENVINO_THROW("Could not open the file: \"", ov::util::wstring_to_string(file_path), '"');
    };

    auto model_proto = parse_from_istream(file_stream);
    file_stream.close();
    return model_proto;
}
#endif

ModelProto parse_from_istream(std::istream& model_stream) {
    if (!model_stream.good()) {
        model_stream.clear();
        model_stream.seekg(0);
        if (!model_stream.good()) {
            OPENVINO_THROW("Provided input stream has incorrect state.");
        }
    }

    ModelProto model_proto;
    if (!model_proto.ParseFromIstream(&model_stream)) {
        OPENVINO_THROW("Error during import of ONNX model provided as input stream "
                       " with binary protobuf message.");
    }

    return model_proto;
}

}  // namespace common
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
