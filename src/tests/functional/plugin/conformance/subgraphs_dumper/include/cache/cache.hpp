// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <memory>

#include "openvino/core/model.hpp"

#include "op_conformance_utils/meta_info/meta_info.hpp"
#include "matchers/single_op/manager.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class ICache {
public:
    std::string m_cache_subdir = ".";

    virtual void update_cache(const std::shared_ptr<ov::Model>& model,
                              const std::string& source_model, bool extract_body = true, bool from_cache = false) {};
    virtual void serialize_cache() {};
    virtual void reset_cache() {};

    void set_serialization_dir(const std::string& serialization_dir) {
        m_serialization_dir = serialization_dir;
    }

    bool is_model_large_to_read(const std::shared_ptr<ov::Model>& model, const std::string& model_path) {
        // ov::Model + ov::CompiledModel
        auto model_bytesize = model->get_graph_size();
        if (2 * model_bytesize >= mem_size) {
            auto model_bytesize_gb = model_bytesize;
            model_bytesize_gb >>= 30;
            auto mem_size_gb = mem_size;
            mem_size_gb >>= 30;
            std::cout << "[ WARNING ] Model " << model_path << " bytesize is " << model_bytesize_gb <<
            "is larger than RAM size: " << mem_size_gb << ". Model will be skipped!" << std::endl;
            return true;
        }
        return false;
    }

    bool is_model_large_to_store_const(const std::shared_ptr<ov::Model>& model) {
        auto model_bytesize = model->get_graph_size();
        size_t gb_8 = 1;
#ifdef OPENVINO_ARCH_64_BIT
        gb_8 <<= 33;
#else
        gb_8 = 0xFFFFFFFFU;
#endif
        if (mem_size <= model_bytesize * 4 || model_bytesize >= gb_8) {
            return true;
        }
        return false;
    }

protected:
    size_t m_serialization_timeout = 60;
    std::string m_serialization_dir = ".";
    static size_t mem_size;

    ICache() = default;

    bool serialize_model(const std::pair<std::shared_ptr<ov::Model>, ov::conformance::MetaInfo>& graph_info,
                         const std::string& rel_serialization_path);
};
}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov