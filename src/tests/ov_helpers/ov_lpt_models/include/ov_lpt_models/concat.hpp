// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class ConcatFunction {
public:
    static std::shared_ptr<ov::Model> get(const ov::element::Type inputPrecision,
                                          const ov::element::Type deqPrecision,
                                          const std::vector<ov::PartialShape>& inputShapes,
                                          const std::vector<DequantizationOperations>& dequantizationsBefore,
                                          const std::int64_t concatAxis,
                                          const ov::element::Type precisionAfter = ov::element::dynamic,
                                          const DequantizationOperations& dequantizationAfter = {});

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const std::shared_ptr<ov::opset1::Constant>& input_constant1,
        const FakeQuantizeOnData& fakeQuantize1,
        const DequantizationOperations& dequantization1,
        const std::shared_ptr<ov::opset1::Constant>& input_constant2,
        const FakeQuantizeOnData& fakeQuantize2,
        const DequantizationOperations& dequantization2);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const FakeQuantizeOnDataWithConstant& fakeQuantize1,
        const FakeQuantizeOnDataWithConstant& fakeQuantize2);

    static std::shared_ptr<ov::Model> getOriginalWithChildAndOutput(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fakeQuantize1,
        const FakeQuantizeOnData& fakeQuantize2);

    static std::shared_ptr<ov::Model> getOriginalWithNeighbors(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const FakeQuantizeOnData& fqOnData3,
        const std::string& neighborType,
        const std::string& additionalLayer);

    static std::shared_ptr<ov::Model> getOriginalWithIntermediate(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ov::Model> getOriginalWithIntermediateAvgPool(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ov::Model> getOriginalWithSplitedIntermediate(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const bool addConvolution);

    static std::shared_ptr<ov::Model> getOriginalSelectionWithIntermediate(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ov::Model> getOriginalWithStridedSlice(
        const ov::element::Type precision,
        const ov::PartialShape inputShape,
        const FakeQuantizeOnData& fq1,
        const FakeQuantizeOnData& fq2,
        const bool ssBeforeConcat,
        const bool ssAfterConcat);

    static std::shared_ptr<ov::Model> getOriginalWithDifferentPrecisionOnChildren(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const std::int64_t axis,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ov::Model> getOriginalWithIntermediateWithConstant(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ov::Model> getOriginalWithReshapeAtTheEndTransformation(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData1,
        const FakeQuantizeOnDataWithConstant& fqOnData2,
        const FakeQuantizeOnDataWithConstant& fqOnData3);

    static std::shared_ptr<ov::Model> getOriginalWithIntermediateReshape(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::Shape& reshapeOutputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type dequantizationPrecision,
        const ov::element::Type precisionBefore,
        const std::vector<ov::PartialShape>& inputShapes,
        const std::vector<DequantizationOperations>& dequantizationsBefore,
        const ov::element::Type precisionAfter,
        const DequantizationOperations& dequantizationAfter,
        const std::int64_t concatAxis);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantize1,
        const FakeQuantizeOnData& fakeQuantize2,
        const DequantizationOperations& dequantizationOperations);

    static std::shared_ptr<ov::Model> get(
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fakeQuantize1,
        const DequantizationOperations::Convert& convert1,
        const DequantizationOperations& dequantization1,
        const FakeQuantizeOnDataWithConstant& fakeQuantize2,
        const DequantizationOperations::Convert& convert2,
        const DequantizationOperations& dequantization2,
        const std::vector<ov::Any>& concatAttributes,
        const ov::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter,
        const std::int64_t& axis,
        const bool addNotPrecisionPreservedOperation = false);

    static std::shared_ptr<ov::Model> get(
        const ov::element::Type inputPrecision,
        const ov::Shape& inputShape1,
        const FakeQuantizeOnDataWithConstant& fakeQuantize1,
        const DequantizationOperations::Convert& convert1,
        const DequantizationOperations& dequantization1,
        const bool addReshape1,
        const ov::Shape& inputShape2,
        const FakeQuantizeOnDataWithConstant& fakeQuantize2,
        const DequantizationOperations::Convert& convert2,
        const DequantizationOperations& dequantization2,
        const bool addReshape2,
        const std::vector<ov::Any>& concatAttributes,
        const ov::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter,
        const std::int64_t& axis,
        const bool addNotPrecisionPreservedOperation = false);

    static std::shared_ptr<ov::Model> getReferenceWithNeighbors(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const FakeQuantizeOnData& fqOnData3,
        const ov::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2,
        const std::string& neighborType,
        const std::string& additionalLayer);

    // TODO: refactor: dequantizationBefore2 <=> dequantizationOperations2
    static std::shared_ptr<ov::Model> getReferenceWithIntermediate(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ov::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationOperations2,
        const ov::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationBefore2);

    static std::shared_ptr<ov::Model> getReferenceWithIntermediateAvgPool(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ov::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationBefore2,
        const ov::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ov::Model> getReferenceWithSplitedIntermediate(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ov::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationBefore2,
        const ov::element::Type precisionAfterOperation,
        const bool addConvolution,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ov::Model> getReferenceSelectionWithIntermediate(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ov::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationBefore2,
        const ov::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ov::Model> getReferenceWithStridedSlice(
        const ov::element::Type inputPrecision,
        const ov::PartialShape inputShape,
        const FakeQuantizeOnData& fq1,
        const FakeQuantizeOnData& fq2,
        const DequantizationOperations& deqBefore,
        const ov::element::Type precisionBeforeConcat,
        const ov::element::Type precisionAfterConcat,
        const bool ssBeforeConcat,
        const bool ssAfterConcat,
        const DequantizationOperations& deqAfter1,
        const DequantizationOperations& deqAfter2);

    static std::shared_ptr<ov::Model> getReferenceWithDifferentPrecisionOnChildren(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const bool multiChannel,
        const std::int64_t axis,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ov::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationBefore2,
        const ov::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter1,
        const DequantizationOperations& dequantizationAfter2);

    static std::shared_ptr<ov::Model> getReferenceWithIntermediateWithConstant(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ov::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter,
        const ov::element::Type precisionAfterDequantization);

    static std::shared_ptr<ov::Model> getReferenceWithReshapeAtTheEndTransformation(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData1,
        const FakeQuantizeOnDataWithConstant& fqOnData2,
        const FakeQuantizeOnDataWithConstant& fqOnData3,
        const ov::element::Type precisionBeforeOp,
        const ov::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations);

    static std::shared_ptr<ov::Model> getReferenceWithIntermediateReshape(
            const ov::element::Type precision,
            const ov::Shape& inputShape,
            const ov::Shape& reshapeOutputShape,
            const FakeQuantizeOnData& fqOnData1,
            const FakeQuantizeOnData& fqOnData2,
            const DequantizationOperations& dequantizationAfter);

private:
    static std::shared_ptr<Node> makeMaxPool(const ov::Output<Node>& parent, const std::vector<size_t>& kernel);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
