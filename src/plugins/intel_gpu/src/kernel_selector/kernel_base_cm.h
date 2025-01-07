// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base.h"

namespace kernel_selector {

class KernelBaseCM : public KernelBase {
public:
    using KernelBase::KernelBase;
    virtual ~KernelBaseCM() {}

protected:
    virtual bool Validate(const Params&) const {
        return true;
    }
    std::shared_ptr<KernelString> GetKernelString(const std::string& kernel_name,
                                                  const std::pair<std::string, std::string>& jit,
                                                  const std::string& entry_point) const {
        std::shared_ptr<KernelString> kernel_string = std::make_shared<KernelString>();

        bool is_cm = true;
        auto codes = db.get(kernel_name, is_cm);

        if (codes.size()) {
            kernel_string->str = codes[0];
            kernel_string->jit = "#include <cm/cm.h>\n#include <cm/cmtl.h>\n";
            kernel_string->jit += jit.first;
            kernel_string->undefs = jit.second;
            kernel_string->options = " -cmc ";

            kernel_string->entry_point = entry_point;
            kernel_string->batch_compilation = true;
            kernel_string->language = KernelLanguage::CM;
        }

        return kernel_string;
    }
};
}  // namespace kernel_selector
