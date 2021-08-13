#include "ngraph/static_dimension.hpp"

using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& str, const StaticDimension& dimension) {
    return str << dimension.get_length();
}
