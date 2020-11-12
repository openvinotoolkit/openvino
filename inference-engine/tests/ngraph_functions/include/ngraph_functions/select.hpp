// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <numeric>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/runtime/reference/select.hpp>

namespace NGraphFunctions {

class Select {
public:
    enum { CONDITION, THEN, ELSE, numOfInputs };
    std::shared_ptr<ngraph::Function> fnPtr;

    Select() = default;

    explicit Select(ngraph::element::Type inType, const std::vector<std::vector<size_t>> &inputShapes, ngraph::op::AutoBroadcastSpec broadcast);

    template<typename outType>
    std::vector<outType> RefImpl(const std::vector<const outType*> &inData, const std::vector<std::vector<size_t>> &inDataShapes,
                                                                                                                const std::vector<size_t> &outputShapes) {
        size_t outElementsCount = std::accumulate(begin(outputShapes), end(outputShapes), 1, std::multiplies<size_t>());

        std::vector<ngraph::Shape> shapes;
        for (auto shape : inDataShapes)
            shapes.push_back(ngraph::Shape(shape));

        size_t maskElementsCount = std::accumulate(begin(inDataShapes[CONDITION]), end(inDataShapes[CONDITION]), 1, std::multiplies<size_t>());
        std::vector<char> mask(maskElementsCount);
        for (size_t i = 0; i < maskElementsCount; i++)
            mask[i] = static_cast<char>(inData[CONDITION][i]);

        std::vector<outType> dstData(outElementsCount);
        ngraph::runtime::reference::select<outType>(mask.data(), inData[THEN], inData[ELSE], dstData.data(), shapes[CONDITION], shapes[THEN], shapes[ELSE],
                                                                                                                                            broadcastType);

        return dstData;
    }

private:
    ngraph::op::AutoBroadcastSpec broadcastType;
};

}  // namespace NGraphFunctions
