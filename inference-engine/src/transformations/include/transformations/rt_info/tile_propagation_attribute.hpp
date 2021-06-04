// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>


namespace ngraph {

class TRANSFORMATIONS_API TileElement {
public:
    int64_t h_begin, h_end, w_begin, w_end;
    std::vector<std::pair<int64_t, int64_t>> m_coord;
    std::function<void(std::shared_ptr<Node>)> modifier = nullptr;

    TileElement() = default;

    TileElement(std::vector<int64_t> slice) {
        assert(slice.size() % 2 == 0);
        for (size_t i = 0; i < slice.size(); i += 2) {
            m_coord.emplace_back(slice[i], slice[i+1]);
        }
        h_begin = slice[0];
        h_end = slice[1];
        w_begin = slice[2];
        w_end = slice[3];
    }

    size_t size() const {
        size_t size{1};
        for (const auto & p : m_coord) {
            size *= p.second - p.first + 1;
        }
        return size;
    }
};

class TRANSFORMATIONS_API Tiles {
public:
    std::vector<int64_t> m_spatial_dims_tiles_count;
    std::vector<int64_t> m_offsets;

    using Tiles_t = std::vector<TileElement>;
    std::vector<TileElement> tiles;

    Tiles(std::vector<int64_t> tiles_count)
        : m_spatial_dims_tiles_count(std::move(tiles_count)) {
        auto tiles_size = std::accumulate(m_spatial_dims_tiles_count.begin(),
                                          m_spatial_dims_tiles_count.end(),
                                          1,
                                          std::multiplies<int64_t>());
        tiles.resize(tiles_size);

        // calculate offsets for efficient indexing
        m_offsets.resize(m_spatial_dims_tiles_count.size());
        for (size_t i = 1; i < m_offsets.size(); ++i) {
            m_offsets[i] = std::accumulate(m_spatial_dims_tiles_count.begin() + i,
                                           m_spatial_dims_tiles_count.end(),
                                           1,
                                           std::multiplies<int64_t>());
        }
    }

    Tiles_t::iterator begin() { return tiles.begin(); }

    Tiles_t ::iterator end() { return tiles.end(); }

    Tiles_t ::const_iterator begin() const { return tiles.begin(); }

    Tiles_t ::const_iterator end() const { return tiles.end(); }

    TileElement & operator[](const size_t key) { return tiles[key]; }

    TileElement at(const size_t key) const { return tiles.at(key); }

    TileElement & at(const std::vector<int64_t> & indices) {
        return tiles[get_offset(indices)];
    }

    TileElement at(const std::vector<int64_t> & indices) const {
        return tiles.at(get_offset(indices));
    }

    size_t get_offset(const std::vector<int64_t> & indices) const {
        assert(indices.size() == m_spatial_dims_tiles_count.size());
        size_t offset{0};
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] > 0) {
                if (i == m_spatial_dims_tiles_count.size() - 1) {
                    offset += indices[i];
                } else {
                    offset += m_offsets[i + 1] * indices[i];
                }
            }
        }
        return offset;
    }

    const std::vector<int64_t> & get_tiles_sizes() const { return m_spatial_dims_tiles_count; }

    size_t size() const { return tiles.size(); }
};

extern template class TRANSFORMATIONS_API VariantImpl<Tiles>;

template<>
class TRANSFORMATIONS_API VariantWrapper<Tiles> : public VariantImpl<Tiles> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::RuntimeAttribute::Tiles", 0};

    const VariantTypeInfo &get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}
};

TRANSFORMATIONS_API Tiles get_tiles(const Input<Node> & input);

TRANSFORMATIONS_API bool has_tiles(const Input<Node> & input);

TRANSFORMATIONS_API void set_tiles(Input<Node> input, const Tiles & tiles);
}  // namespace ngraph
