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

template class DirectValueAccessor<std::vector<std::string>>;
template class DirectValueAccessor<std::vector<signed char>>;
template class DirectValueAccessor<std::vector<unsigned short>>;
template class DirectValueAccessor<std::vector<unsigned int>>;
template class DirectValueAccessor<std::vector<unsigned char>>;
template class DirectValueAccessor<std::vector<short>>;
template class DirectValueAccessor<std::vector<double>>;
template class DirectValueAccessor<std::vector<float>>;
template class DirectValueAccessor<std::vector<int>>;
template class DirectValueAccessor<std::vector<long>>;
template class DirectValueAccessor<std::vector<unsigned long>>;
template class DirectValueAccessor<std::set<std::string>>;

template class EnumAttributeAdapterBase<op::PadMode>;
template class EnumAttributeAdapterBase<op::FillMode>;
template class EnumAttributeAdapterBase<op::PadType>;
template class EnumAttributeAdapterBase<op::RoundingType>;
template class EnumAttributeAdapterBase<op::AutoBroadcastType>;
template class EnumAttributeAdapterBase<op::BroadcastType>;
template class EnumAttributeAdapterBase<op::EpsMode>;
template class EnumAttributeAdapterBase<op::TopKSortType>;
template class EnumAttributeAdapterBase<op::TopKMode>;
template class EnumAttributeAdapterBase<op::PhiloxAlignment>;
template class EnumAttributeAdapterBase<op::RecurrentSequenceDirection>;
template class EnumAttributeAdapterBase<element::Type_t>;
}  // namespace ov
