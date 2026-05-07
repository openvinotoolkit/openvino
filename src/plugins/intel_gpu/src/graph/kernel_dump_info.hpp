// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace cldnn {

class KernelDumpInfo {
public:
    KernelDumpInfo() = default;

    void add_entry_point(const std::string& entry_point) {
        if (!m_entries.empty()) {
            m_entries += " ";
        }
        m_entries += entry_point;
    }

    void set_batch_hash(const std::string& batch_hash) {
        m_batch_hash = batch_hash;
    }

    void clear_entries() {
        m_entries.clear();
    }

    const std::string& get_batch_hash() const {
        return m_batch_hash;
    }

    const std::string& get_entries() const {
        return m_entries;
    }

    bool has_entries() const {
        return !m_entries.empty();
    }

private:
    std::string m_batch_hash;
    std::string m_entries;
};

}  // namespace cldnn
