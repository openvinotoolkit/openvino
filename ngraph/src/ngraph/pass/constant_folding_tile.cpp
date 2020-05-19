//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "constant_folding.hpp"
#include "ngraph/op/tile.hpp"
#include "ngraph/runtime/reference/tile.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static shared_ptr<op::Constant> fold_constant_tile(const shared_ptr<op::Constant>& data,
                                                   const shared_ptr<Node>& tile)
{
    runtime::AlignedBuffer buffer(shape_size(tile->get_shape()) * sizeof(T));
    T* data_ptr = buffer.get_ptr<T>();
    // No need to call the reference kernel.
    if (shape_size(tile->get_shape()) == 0)
    {
        return make_shared<op::Constant>(
            tile->get_output_element_type(0), tile->get_output_shape(0), data_ptr);
    }

    if (auto tile_v0 = as_type_ptr<op::v0::Tile>(tile))
    {
        runtime::reference::tile<T>(
            data->get_data_ptr<T>(), data_ptr, data->get_shape(), tile_v0->get_shape());
    }
    else
    {
        throw ngraph_error("Unsupported op in tile constant folding.");
    }

    return make_shared<op::Constant>(
        tile->get_output_element_type(0), tile->get_output_shape(0), data_ptr);
}

void pass::ConstantFolding::construct_constant_tile()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 2, 3}, pattern::has_class<op::Constant>());
    auto repeats_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto tile_v0 = make_shared<op::v0::Tile>(data_label, repeats_label);

    auto constant_tile_callback = [data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_tile_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto data = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        auto tile = m.get_match_root();

        NGRAPH_CHECK(revalidate_and_ensure_static(tile));

        std::shared_ptr<Node> replacement;
        auto data_type = data->get_output_element_type(0);
        switch (data_type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_tile_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_tile_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_tile_callback");
            break;
        case element::Type_t::boolean: replacement = fold_constant_tile<char>(data, tile); break;
        case element::Type_t::bf16: replacement = fold_constant_tile<bfloat16>(data, tile); break;
        case element::Type_t::f16: replacement = fold_constant_tile<float16>(data, tile); break;
        case element::Type_t::f32: replacement = fold_constant_tile<float>(data, tile); break;
        case element::Type_t::f64: replacement = fold_constant_tile<double>(data, tile); break;
        case element::Type_t::i8: replacement = fold_constant_tile<int8_t>(data, tile); break;
        case element::Type_t::i16: replacement = fold_constant_tile<int16_t>(data, tile); break;
        case element::Type_t::i32: replacement = fold_constant_tile<int32_t>(data, tile); break;
        case element::Type_t::i64: replacement = fold_constant_tile<int64_t>(data, tile); break;
        case element::Type_t::u8: replacement = fold_constant_tile<uint8_t>(data, tile); break;
        case element::Type_t::u16: replacement = fold_constant_tile<uint16_t>(data, tile); break;
        case element::Type_t::u32: replacement = fold_constant_tile<uint32_t>(data, tile); break;
        case element::Type_t::u64: replacement = fold_constant_tile<uint64_t>(data, tile); break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto tile_matcher_v0 = make_shared<pattern::Matcher>(tile_v0, "ConstantFolding.ConstantTileV0");
    this->add_matcher(tile_matcher_v0, constant_tile_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
