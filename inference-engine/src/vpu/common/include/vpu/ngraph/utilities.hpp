// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/type/element_type.hpp"

namespace vpu {

std::vector<std::int64_t> evaluateTargetShape(const ngraph::Output<ngraph::Node>& value);

std::shared_ptr<ngraph::Node> shapeToConstant(const ngraph::element::Type& type, const ngraph::Shape& shape);

std::shared_ptr<ngraph::Node> gatherShapeElements(const ngraph::Output<ngraph::Node>&, int startIndex, size_t elemCount);

void printTo(std::ostream& stream, const ngraph::NodeTypeInfo& object);

}  // namespace vpu
