// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include <vector>

namespace cldnn {

/// @brief Provides mutable data.
/// @details This primitive allows to pass data which can be written to during training.
/// For example, weights and biases for scoring networks.
/// This primitive can be also set as other primitive's output. In this case the underlying buffer will be the same in mutable_data and preceding primitive.
struct mutable_data : public primitive_base<mutable_data> {
    CLDNN_DECLARE_PRIMITIVE(mutable_data)

    mutable_data() : primitive_base("", {}) {}

    /// @brief Enum type to specify function for data filling.
    enum filler_type { no_fill, zero, one, xavier };

    /// @brief Constructs mutable_data primitive.
    /// @param id This primitive id.
    /// @param mem @ref memory object which contains data.
    /// @param filler_type @ref data filling function, default is zero
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    mutable_data(const primitive_id& id,
                 memory::ptr mem,
                 filler_type fill_type = filler_type::no_fill)
        : primitive_base(id, {}), mem(mem), fill_type(fill_type) {}

    /// @brief Constructs mutable_data primitive with inputs.
    /// @param id This primitive id.
    /// @param input Vector of input primitives ids.
    /// @param mem @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    /// @param filler_type @ref data filling function, default is zero
    mutable_data(const primitive_id& id,
                 const std::vector<input_info>& inputs,
                 memory::ptr mem,
                 filler_type fill_type = filler_type::no_fill)
        : primitive_base(id, inputs), mem(std::move(mem)), fill_type(fill_type) {}

    /// @brief @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    memory::ptr mem;

    /// @brief Specifies function which will be used to fill weights.
    filler_type fill_type = filler_type::no_fill;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, id);
        return seed;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<mutable_data>::save(ob);
        ob << make_data(&fill_type, sizeof(filler_type));

        ob << mem->get_layout();

        const auto _allocation_type = mem->get_allocation_type();
        ob << make_data(&_allocation_type, sizeof(_allocation_type));

        size_t data_size = mem->size();
        ob << make_data(&data_size, sizeof(size_t));

        if (_allocation_type == allocation_type::usm_host || _allocation_type == allocation_type::usm_shared) {
            ob << make_data(mem->buffer_ptr(), data_size);
        } else {
            std::vector<uint8_t> _buf;
            _buf.resize(data_size);
            stream* strm = reinterpret_cast<stream*>(ob.get_stream());
            mem->copy_to(*strm, _buf.data());
            ob << make_data(_buf.data(), data_size);
        }
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<mutable_data>::load(ib);
        ib >> make_data(&fill_type, sizeof(filler_type));

        layout output_layout = layout();
        ib >> output_layout;

        allocation_type _allocation_type = allocation_type::unknown;
        ib >> make_data(&_allocation_type, sizeof(_allocation_type));

        size_t data_size = 0;
        ib >> make_data(&data_size, sizeof(size_t));

        mem = ib.get_engine().allocate_memory(output_layout, _allocation_type, false);

        if (_allocation_type == allocation_type::usm_host || _allocation_type == allocation_type::usm_shared) {
            ib >> make_data(mem->buffer_ptr(), data_size);
        } else {
            std::vector<uint8_t> _buf;
            _buf.resize(data_size);
            ib >> make_data(_buf.data(), data_size);
            // stream* strm = reinterpret_cast<stream*>(ib.get_stream());
            auto& strm = ib.get_engine().get_service_stream();
            mem->copy_from(strm, _buf.data());
        }
    }
};
}  // namespace cldnn
