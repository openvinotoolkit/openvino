// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <functional>

#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API Serialize;
class TRANSFORMATIONS_API StreamSerialize;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Serialize transformation converts ngraph::Function into IR files
 * @attention
 * - dynamic shapes are not supported
 * - order of generated layers in xml file is ngraph specific (given by
 * get_ordered_ops()); MO generates file with different order, but they are
 * logically equivalent
 */
class ngraph::pass::Serialize : public ngraph::pass::FunctionPass {
public:
    enum class Version : uint8_t {
        IR_V10 = 10,            // v10 IR
        IR_V11 = 11             // v11 IR
    };
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

    Serialize(std::ostream& xmlFile, std::ostream& binFile,
              std::map<std::string, ngraph::OpSet> custom_opsets,
              Version version = Version::IR_V11);
    Serialize(std::ostream& xmlFile, std::ostream& binFile,
              Version version = Version::IR_V11);

    Serialize(const std::string& xmlPath, const std::string& binPath,
              std::map<std::string, ngraph::OpSet> custom_opsets,
              Version version = Version::IR_V11);
    Serialize(const std::string& xmlPath, const std::string& binPath,
              Version version = Version::IR_V11);

private:
    std::ostream * m_xmlFile;
    std::ostream * m_binFile;
    const std::string m_xmlPath;
    const std::string m_binPath;
    const Version m_version;
    const std::map<std::string, ngraph::OpSet> m_custom_opsets;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief StreamSerialize transformation converts ngraph::Function into single binary stream
 * @attention
 * - dynamic shapes are not supported
 */
class ngraph::pass::StreamSerialize : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;

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
                    std::map<std::string, ngraph::OpSet> && custom_opsets = {},
                    const std::function<void(std::ostream &)> & custom_data_serializer = {},
                    Serialize::Version version = Serialize::Version::IR_V11);

private:
    std::ostream & m_stream;
    std::map<std::string, ngraph::OpSet> m_custom_opsets;
    std::function<void(std::ostream &)> m_custom_data_serializer;
    const Serialize::Version m_version;
};
