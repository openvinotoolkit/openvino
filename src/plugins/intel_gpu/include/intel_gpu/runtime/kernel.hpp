// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_args.hpp"
#include "event.hpp"

#include <memory>
#include <vector>

namespace cldnn {

using kernel_id = std::string;

class kernel {
public:
    using ptr = std::shared_ptr<kernel>;
    virtual std::shared_ptr<kernel> clone() const = 0;
    virtual ~kernel() = default;
    virtual std::string get_id() const { return ""; }

#ifdef GPU_DEBUG_CONFIG
    struct kernel_properties {
        uint64_t local_mem_size = 0;
        uint64_t private_mem_size = 0;
        uint64_t spill_mem_size = 0;

        std::string to_string() {
            std::stringstream ss;

            ss << "SLM=" << local_mem_size << " "
               << "SPILL=" << spill_mem_size << " "
               << "TPM=" << private_mem_size;

            return ss.str();
        }
    };

    virtual kernel_properties get_properties() const = 0;
#endif
};

}  // namespace cldnn
