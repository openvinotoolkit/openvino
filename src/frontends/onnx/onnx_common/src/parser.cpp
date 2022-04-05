// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_common/parser.hpp"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <onnx/onnx_pb.h>

#include <ngraph/file_util.hpp>

#include "ngraph/except.hpp"

namespace ngraph {
namespace onnx_common {
ONNX_NAMESPACE::ModelProto parse_from_file(const std::string& file_path) {
    std::ifstream file_stream{file_path, std::ios::in | std::ios::binary};

    if (!file_stream.is_open()) {
        throw ngraph_error("Could not open the file: " + file_path);
    };

    auto model_proto = parse_from_istream(file_stream);
    file_stream.close();
    return model_proto;
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
ONNX_NAMESPACE::ModelProto parse_from_file(const std::wstring& file_path) {
    std::ifstream file_stream{file_path, std::ios::in | std::ios::binary};

    if (!file_stream.is_open()) {
        NGRAPH_SUPPRESS_DEPRECATED_START
        throw ngraph_error("Could not open the file: " + file_util::wstring_to_string(file_path));
        NGRAPH_SUPPRESS_DEPRECATED_END
    };

    auto model_proto = parse_from_istream(file_stream);
    file_stream.close();
    return model_proto;
}
#endif

ONNX_NAMESPACE::ModelProto parse_from_istream(std::istream& model_stream) {
    if (!model_stream.good()) {
        model_stream.clear();
        model_stream.seekg(0);
        if (!model_stream.good()) {
            throw ngraph_error("Provided input stream has incorrect state.");
        }
    }

    ONNX_NAMESPACE::ModelProto model_proto;
    if (!model_proto.ParseFromIstream(&model_stream)) {
        throw ngraph_error("Error during import of ONNX model provided as input stream "
                           " with binary protobuf message.");
    }

    return model_proto;
}
}  // namespace onnx_common
}  // namespace ngraph
