#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace runtime {
namespace interpreter {
namespace pass {

// Transformation inserts Convert on first Multinomial's input if input type is on 'ov::util::unsupported_types()' list.
// Input type and output type of this Convert is the same (hence 'nop' in the transformation's name) and the Convert is
// marked with keep_convert_precision attribute so ConvertPrecision doesn't eliminate it.
// No-op Convert is required to keep the Multinomial with original precision. Multinomial relies on RandomUniform and
// its output can be very different (not just by a small margin) with different types (f32 vs. f16 for example).
class InsertNopConvertOnMultinomial : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertNopConvertOnMultinomial", "0");
    InsertNopConvertOnMultinomial();
};

}  // namespace pass
}  // namespace interpreter
}  // namespace runtime
}  // namespace ov
