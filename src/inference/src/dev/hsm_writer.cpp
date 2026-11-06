// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

namespace {

class WriteErrorCategory final : public std::error_category {
public:
    const char* name() const noexcept override {
        return "ov::runtime::hsm::Write";
    }

    std::string message(int ev) const override {
        switch (static_cast<WriteErrc>(ev)) {
        case WriteErrc::write_failed:
            return "the destination ran out of room, or a write failed";
        default:
            return "unknown error";
        }
    }
};

}  // namespace

std::error_code make_error_code(WriteErrc e) noexcept {
    static const WriteErrorCategory category;
    return {static_cast<int>(e), category};
}

void IWriter::add_sections(const std::vector<ISectionWriterHandler*>& handlers) {
    for (const auto* handler : handlers) {
        if (handler != nullptr) {
            handler->handle_section(*this);
        }
    }
}

}  // namespace v1
}  // namespace ov::runtime::hsm
