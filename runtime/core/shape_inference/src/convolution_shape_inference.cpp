
#include <convolution_shape_inference.hpp>

namespace ov {
namespace op {
namespace v1 {

int64_t calculate_num_spatial(const Convolution* op,
                              const PartialShape& input_shape,
                              const PartialShape& filters_shape,
                              const int64_t& num_non_spatial_data_dims,
                              const int64_t& num_non_spatial_filter_dims) {
    int64_t num_spatial = op->m_num_spatial;
    if (num_spatial == -1) {
        const auto &input_rank = input_shape.rank();
        const auto &filters_rank = filters_shape.rank();

        if (const auto &size = op->m_dilations.size())
            num_spatial = static_cast<int64_t>(size);
        if (const auto &size = op->m_strides.size())
            num_spatial = static_cast<int64_t>(size);
        if (const auto &size = op->m_pads_begin.size())
            num_spatial = static_cast<int64_t>(size);
        if (const auto &size = op->m_pads_end.size())
            num_spatial = static_cast<int64_t>(size);
        if (input_rank.is_static())
            num_spatial = input_rank.get_length() - num_non_spatial_data_dims;
        if (filters_rank.is_static())
            num_spatial = filters_rank.get_length() - num_non_spatial_filter_dims;
    }
    return num_spatial;
}

void update_and_validate_attributes(Convolution* op) {
    const auto& num_spatial = op->m_num_spatial;
    if (num_spatial != -1) {
        auto& strides = op->m_strides;
        auto& dilations = op->m_dilations;
        auto& pad_begin = op->m_pads_begin;
        auto& pad_end = op->m_pads_end;
        auto& auto_pad = op->m_auto_pad;

        if (strides.empty())
            strides = Strides(num_spatial, 1);
        if (dilations.empty())
            dilations = Strides(num_spatial, 1);
        if (pad_begin.empty() || auto_pad == op::PadType::VALID)
            pad_begin = CoordinateDiff(num_spatial, 0);
        if (pad_end.empty() || auto_pad == op::PadType::VALID)
            pad_end = CoordinateDiff(num_spatial, 0);

        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(strides.size()) == num_spatial,
                              "Strides should be defined for all and only spatial features.");
        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(dilations.size()) == num_spatial,
                              "Dilations should be defined for all and only spatial features.");
        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(pad_begin.size()) == num_spatial &&
                              static_cast<int64_t>(pad_end.size()) == num_spatial,
                              "Pads should be defined for all and only spatial features.");
        NODE_VALIDATION_CHECK(op,
                              std::all_of(dilations.begin(),
                                          dilations.end(),
                                          [](const size_t &i) {
                                              return i > 0;
                                          }),
                              "Filter dilation (",
                              dilations,
                              ") has zero dimension.");
        NODE_VALIDATION_CHECK(op,
                              std::all_of(strides.begin(),
                                          strides.end(),
                                          [](const size_t &i) {
                                              return i > 0;
                                          }),
                              "Filter strides (",
                              strides,
                              ") has zero dimension.");
    }
}

}
}
}