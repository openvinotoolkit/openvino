// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/attribute_adapter.hpp"

namespace ov {

ValueAccessor<void>::~ValueAccessor() = default;

AttributeAdapter<float>::~AttributeAdapter() = default;
AttributeAdapter<double>::~AttributeAdapter() = default;
AttributeAdapter<std::string>::~AttributeAdapter() = default;
AttributeAdapter<bool>::~AttributeAdapter() = default;
AttributeAdapter<int8_t>::~AttributeAdapter() = default;
AttributeAdapter<int16_t>::~AttributeAdapter() = default;
AttributeAdapter<int32_t>::~AttributeAdapter() = default;
AttributeAdapter<int64_t>::~AttributeAdapter() = default;
AttributeAdapter<uint8_t>::~AttributeAdapter() = default;
AttributeAdapter<uint16_t>::~AttributeAdapter() = default;
AttributeAdapter<uint32_t>::~AttributeAdapter() = default;
AttributeAdapter<uint64_t>::~AttributeAdapter() = default;
#if defined(__APPLE__) || defined(__EMSCRIPTEN__)
AttributeAdapter<size_t>::~AttributeAdapter() = default;
AttributeAdapter<std::vector<size_t>>::~AttributeAdapter() = default;
#endif
AttributeAdapter<std::vector<int8_t>>::~AttributeAdapter() = default;
AttributeAdapter<std::vector<int16_t>>::~AttributeAdapter() = default;
AttributeAdapter<std::vector<int32_t>>::~AttributeAdapter() = default;
AttributeAdapter<std::vector<int64_t>>::~AttributeAdapter() = default;
AttributeAdapter<std::vector<uint8_t>>::~AttributeAdapter() = default;
AttributeAdapter<std::vector<uint16_t>>::~AttributeAdapter() = default;
AttributeAdapter<std::vector<uint32_t>>::~AttributeAdapter() = default;
AttributeAdapter<std::vector<uint64_t>>::~AttributeAdapter() = default;
AttributeAdapter<std::vector<float>>::~AttributeAdapter() = default;
AttributeAdapter<std::vector<double>>::~AttributeAdapter() = default;
AttributeAdapter<std::vector<std::string>>::~AttributeAdapter() = default;
AttributeAdapter<std::set<std::string>>::~AttributeAdapter() = default;

}  // namespace ov
