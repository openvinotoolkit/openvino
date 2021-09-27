// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>

#include "openvino/opsets/opset.hpp"
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
 */
class OPENVINO_API Serialize : public ov::pass::FunctionPass {
public:
    OPENVINO_RTTI("Serialize");

    enum class Version { IR_V10 };
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

    Serialize(std::ostream& xmlFile,
              std::ostream& binFile,
              Version version = Version::IR_V10,
              std::map<std::string, ov::OpSet> custom_opsets = {});

    Serialize(const std::string& xmlPath,
              const std::string& binPath,
              Version version = Version::IR_V10,
              std::map<std::string, ov::OpSet> custom_opsets = {});

private:
    std::ostream* m_xmlFile;
    std::ostream* m_binFile;
    const std::string m_xmlPath;
    const std::string m_binPath;
    const Version m_version;
    const std::map<std::string, ov::OpSet> m_custom_opsets;
};

/**
 * @brief StreamSerialize transformation converts ngraph::Function into single binary stream
 * @attention
 * - dynamic shapes are not supported
 */
class OPENVINO_API StreamSerialize : public ov::pass::FunctionPass {
public:
    OPENVINO_RTTI("Serialize");

    struct DataHeader {
        size_t custom_data_offset;
        size_t custom_data_size;
        size_t consts_offset;
        size_t consts_size;
        size_t model_offset;
        size_t model_size;
    };

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

    StreamSerialize(std::ostream& stream,
                    std::map<std::string, ov::OpSet>&& custom_opsets = {},
                    const std::function<void(std::ostream&)>& custom_data_serializer = {},
                    Serialize::Version version = Serialize::Version::IR_V10);

private:
    std::ostream& m_stream;
    std::map<std::string, ov::OpSet> m_custom_opsets;
    std::function<void(std::ostream&)> m_custom_data_serializer;
};

}  // namespace pass
}  // namespace ov
