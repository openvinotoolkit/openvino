// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ConcatFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantize1,
        const FakeQuantizeOnData& fakeQuantize2);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnDataWithConstant& fakeQuantize1,
        const FakeQuantizeOnDataWithConstant& fakeQuantize2);

    static std::shared_ptr<ngraph::Function> getOriginalWithChildAndOutput(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantize1,
        const FakeQuantizeOnData& fakeQuantize2);

    static std::shared_ptr<ngraph::Function> getOriginalWithNeighbors(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const FakeQuantizeOnData& fqOnData3);

    static std::shared_ptr<ngraph::Function> getOriginalWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginalWithSplitedIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginalSelectionWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginalWithStridedSlice(
        const ngraph::element::Type precision,
        const ngraph::Shape inputShape,
        const FakeQuantizeOnData& fq1,
        const FakeQuantizeOnData& fq2,
        const bool ssBeforeConcat,
        const bool ssAfterConcat);

    static std::shared_ptr<ngraph::Function> getOriginalWithDifferentPrecisionOnChilds(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginalWithIntermediateWithConstant(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginalWithReshapeAtTheEndTransformation(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData1,
        const FakeQuantizeOnDataWithConstant& fqOnData2,
        const FakeQuantizeOnDataWithConstant& fqOnData3);

    static std::shared_ptr<ngraph::Function> getOriginalWithIntermediateReshape(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::Shape& reshapeOutputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantize1,
        const FakeQuantizeOnData& fakeQuantize2,
        const DequantizationOperations& dequantizationOperations);

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnDataWithConstant& fakeQuantize1,
        const DequantizationOperations::Convert& convert1,
        const DequantizationOperations& dequantization1,
        const FakeQuantizeOnDataWithConstant& fakeQuantize2,
        const DequantizationOperations::Convert& convert2,
        const DequantizationOperations& dequantization2,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter);

    static std::shared_ptr<ngraph::Function> getReferenceWithNeighbors(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const FakeQuantizeOnData& fqOnData3,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ngraph::Function> getReferenceWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationBefore2,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ngraph::Function> getReferenceWithSplitedIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationBefore2,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ngraph::Function> getReferenceSelectionWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationBefore2,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ngraph::Function> getReferenceWithStridedSlice(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape inputShape,
        const FakeQuantizeOnData& fq1,
        const FakeQuantizeOnData& fq2,
        const DequantizationOperations& deqBefore,
        const ngraph::element::Type precisionBeforeConcat,
        const ngraph::element::Type precisionAfterConcat,
        const bool ssBeforeConcat,
        const bool ssAfterConcat,
        const DequantizationOperations& deqAfter1,
        const DequantizationOperations& deqAfter2);

    static std::shared_ptr<ngraph::Function> getReferenceWithDifferentPrecisionOnChilds(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool multiChannel,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter1,
        const DequantizationOperations& dequantizationAfter2);

    static std::shared_ptr<ngraph::Function> getReferenceWithIntermediateWithConstant(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter,
        const ngraph::element::Type precisionAfterDequantization);

    static std::shared_ptr<ngraph::Function> getReferenceWithReshapeAtTheEndTransformation(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData1,
        const FakeQuantizeOnDataWithConstant& fqOnData2,
        const FakeQuantizeOnDataWithConstant& fqOnData3,
        const ngraph::element::Type precisionBeforeOp,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations);

    static std::shared_ptr<ngraph::Function> getReferenceWithIntermediateReshape(
            const ngraph::element::Type precision,
            const ngraph::Shape& inputShape,
            const ngraph::Shape& reshapeOutputShape,
            const FakeQuantizeOnData& fqOnData1,
            const FakeQuantizeOnData& fqOnData2,
            const DequantizationOperations& dequantizationAfter);

private:
    static std::shared_ptr<Node> makeMaxPool(const Output<Node>& parent, const std::vector<size_t>& kernel);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
