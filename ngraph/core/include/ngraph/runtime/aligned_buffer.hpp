// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        /// \brief Allocates a block of memory on the specified alignment. The actual size of the
        /// allocated memory is larger than the requested size by the alignment, so allocating 1
        /// byte
        /// on 64 byte alignment will allocate 65 bytes.
        class NGRAPH_API AlignedBuffer
        {
        public:
            // Allocator objects and the allocation interfaces are owned by the
            // creators of AlignedBuffers. They need to ensure that the lifetime of
            // allocator exceeds the lifetime of this AlignedBuffer.
            AlignedBuffer(size_t byte_size, size_t alignment = 64);

            AlignedBuffer();
            virtual ~AlignedBuffer();

            AlignedBuffer(AlignedBuffer&& other);
            AlignedBuffer& operator=(AlignedBuffer&& other);

            size_t size() const { return m_byte_size; }
            void* get_ptr(size_t offset) const { return m_aligned_buffer + offset; }
            void* get_ptr() { return m_aligned_buffer; }
            const void* get_ptr() const { return m_aligned_buffer; }
            template <typename T>
            T* get_ptr()
            {
                return reinterpret_cast<T*>(m_aligned_buffer);
            }
            template <typename T>
            const T* get_ptr() const
            {
                return reinterpret_cast<const T*>(m_aligned_buffer);
            }

            template <typename T>
            explicit operator T*()
            {
                return get_ptr<T>();
            }

        private:
            AlignedBuffer(const AlignedBuffer&) = delete;
            AlignedBuffer& operator=(const AlignedBuffer&) = delete;

        protected:
            char* m_allocated_buffer;
            char* m_aligned_buffer;
            size_t m_byte_size;
        };
    } // namespace runtime
    template <>
    class NGRAPH_API AttributeAdapter<std::shared_ptr<runtime::AlignedBuffer>>
        : public DirectValueAccessor<std::shared_ptr<runtime::AlignedBuffer>>
    {
    public:
        AttributeAdapter(std::shared_ptr<runtime::AlignedBuffer>& value);

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<std::shared_ptr<runtime::AlignedBuffer>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
