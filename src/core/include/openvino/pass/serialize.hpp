// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>

#include "ngraph/opsets/opset.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {

/**
 * @brief Serialize transformation converts ngraph::Function into IR files
 * @attention
 * - dynamic shapes are not supported
 * - order of generated layers in xml file is ngraph specific (given by
 * get_ordered_ops()); MO generates file with different order, but they are
 * logically equivalent
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API Serialize : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("Serialize");

    enum class Version : uint8_t {
        UNSPECIFIED = 0,  // Use the latest or function version
        IR_V10 = 10,      // v10 IR
        IR_V11 = 11       // v11 IR
    };
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    OPENVINO_DEPRECATED("This constructor is deprecated. Please use new extension API")
    Serialize(std::ostream& xmlFile,
              std::ostream& binFile,
              std::map<std::string, ngraph::OpSet> custom_opsets,
              Version version = Version::UNSPECIFIED);
    Serialize(std::ostream& xmlFile, std::ostream& binFile, Version version = Version::UNSPECIFIED);

    OPENVINO_DEPRECATED("This constructor is deprecated. Please use new extension API")
    Serialize(const std::string& xmlPath,
              const std::string& binPath,
              std::map<std::string, ngraph::OpSet> custom_opsets,
              Version version = Version::UNSPECIFIED);
    Serialize(const std::string& xmlPath, const std::string& binPath, Version version = Version::UNSPECIFIED);

private:
    std::ostream* m_xmlFile;
    std::ostream* m_binFile;
    const std::string m_xmlPath;
    const std::string m_binPath;
    const Version m_version;
    const std::map<std::string, ngraph::OpSet> m_custom_opsets;
};

/**
 * @brief StreamSerialize transformation converts ngraph::Function into single binary stream
 * @attention
 * - dynamic shapes are not supported
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API StreamSerialize : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("StreamSerialize");

    struct DataHeader {
        size_t custom_data_offset;
        size_t custom_data_size;
        size_t consts_offset;
        size_t consts_size;
        size_t model_offset;
        size_t model_size;
    };

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    OPENVINO_DEPRECATED("This constructor is deprecated. Please use new extension API")
    StreamSerialize(std::ostream& stream,
                    std::map<std::string, ngraph::OpSet>&& custom_opsets = {},
                    const std::function<void(std::ostream&)>& custom_data_serializer = {},
                    Serialize::Version version = Serialize::Version::UNSPECIFIED);
    StreamSerialize(std::ostream& stream,
                    const std::function<void(std::ostream&)>& custom_data_serializer = {},
                    Serialize::Version version = Serialize::Version::UNSPECIFIED);

private:
    std::ostream& m_stream;
    std::map<std::string, ngraph::OpSet> m_custom_opsets;
    std::function<void(std::ostream&)> m_custom_data_serializer;
    const Serialize::Version m_version;
};

}  // namespace pass
}  // namespace ov
