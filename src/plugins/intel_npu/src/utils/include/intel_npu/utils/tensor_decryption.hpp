// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils.hpp"

namespace intel_npu {

namespace utils {

/**
 * @brief Uses the provided decryption callback to decrypt the given payload.
 */
static inline void decrypt_payload(ov::Tensor& payload,
                                   const ov::EncryptionCallbacks& encryption_callbacks,
                                   const Logger& logger) {
    OPENVINO_ASSERT(encryption_callbacks.decrypt, "Decryption requested without providing a decryption callback");

    std::string decryptedBlobStr;
    {
        std::string encryptedBlobStr(payload.data<const char>(), payload.get_byte_size());  // +1x blob size
        decryptedBlobStr = encryption_callbacks.decrypt(encryptedBlobStr);                  // +1x blob size
    }  // -1x blob size when deallocating temporary encrypted blob string
    ov::Allocator customAllocator{utils::AlignedAllocator{utils::STANDARD_PAGE_SIZE}};
    size_t alignedSize = utils::align_size_to_standard_page_size(decryptedBlobStr.size());
    size_t paddingSize = alignedSize - decryptedBlobStr.size();
    payload = ov::Tensor(ov::element::u8, ov::Shape{alignedSize},
                         customAllocator);  // +1x blob size
    std::memcpy(payload.data<char>(), decryptedBlobStr.c_str(), decryptedBlobStr.size());
    if (paddingSize > 0) {
        // The blob obtained after decryption is expected to be the same as the blob we had before encryption.
        // That means blobs compiled with the current plugin version are expected to be already aligned.
        // However, the alignment might not be mandatory in a future plugin version. For this scenario, the
        // padding is added here in order to make use of this "non-copy optimization".
        logger.warning("Decrypted blob size was not page aligned, additional %zu bytes padding will be added",
                       paddingSize);
        std::memset(payload.data<char>() + decryptedBlobStr.size(), 0, paddingSize);
    }
}  // -1x blob size when deallocating decrypted blob string

}  // namespace utils

}  // namespace intel_npu
