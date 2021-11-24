// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "utility.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/variant.hpp"
#include "tensorflow_frontend/frontend.hpp"


namespace ov {
namespace frontend {

class TF_API ConversionExtension : public ov::Extension {
public:
    using Ptr = std::shared_ptr<ConversionExtension>;
    ConversionExtension() = delete;
    ConversionExtension(const std::string& op_type,
                        const FrontEndTF::CreatorFunction& converter) :
            m_op_type(op_type), m_converter(converter) {
    }

    const FrontEndTF::CreatorFunction& get_converter() const { return m_converter; }
    const std::string& get_op_type() const { return m_op_type; }
private:
    std::string m_op_type;
    FrontEndTF::CreatorFunction m_converter;
};

}
}