"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.convertTensor = exports.parseOriginalTensor = exports.convertShape = exports.parseOriginalShape = exports.uploadFile = exports.getFileDataAsArray = exports.isNodeEnv = void 0;
const openvinojs_common_1 = require("openvinojs-common");
const maps_js_1 = require("./maps.js");
function isNodeEnv() {
    return typeof window === 'undefined';
}
exports.isNodeEnv = isNodeEnv;
async function getFileDataAsArray(path) {
    const fileData = isNodeEnv()
        ? await getFileDataNode(path)
        : await getFileDataBrowser(path);
    if (!fileData)
        throw new Error(`File '${path}' cannot be loaded!`);
    return new Uint8Array(fileData);
}
exports.getFileDataAsArray = getFileDataAsArray;
async function getFileDataNode(path) {
    const { readFileSync } = await import('node:fs');
    return readFileSync(path);
}
async function getFileDataBrowser(path) {
    const blob = await fetch(path).then(response => !response.ok ? response.blob() : null);
    return blob ? await blob.arrayBuffer() : null;
}
function uploadFile(ov, filename, data) {
    const stream = ov.FS.open(filename, 'w+');
    ov.FS.write(stream, data, 0, data.length, 0);
    ov.FS.close(stream);
}
exports.uploadFile = uploadFile;
const SHAPE_HEAP = maps_js_1.heapLabelByArrayTypeMap[openvinojs_common_1.Shape.TYPE.name];
function parseOriginalShape(ov, originalShape) {
    const originalDim = originalShape.getDim();
    const originalDataPointer = originalShape.getData();
    const dimensions = new openvinojs_common_1.Shape.TYPE(originalDim);
    for (let i = 0; i < originalDim; i++) {
        const dimension = ov[SHAPE_HEAP][originalDataPointer / openvinojs_common_1.Shape.TYPE.BYTES_PER_ELEMENT + i];
        dimensions[i] = dimension;
    }
    return new openvinojs_common_1.Shape(...dimensions);
}
exports.parseOriginalShape = parseOriginalShape;
function convertShape(ov, shape) {
    const originalDimensions = new openvinojs_common_1.Shape.TYPE(shape.data);
    const elementSizeInBytes = originalDimensions.BYTES_PER_ELEMENT;
    const heapSpace = ov._malloc(originalDimensions.length * elementSizeInBytes);
    const offset = Math.sqrt(elementSizeInBytes);
    ov[SHAPE_HEAP].set(originalDimensions, heapSpace >> offset);
    return { obj: new ov.Shape(heapSpace, shape.dim), free: () => ov._free(heapSpace) };
}
exports.convertShape = convertShape;
function parseOriginalTensor(ov, originalTensor) {
    const precision = maps_js_1.ovTypesMap[originalTensor.getPrecision()];
    const shape = parseOriginalShape(ov, originalTensor.getShape());
    const dataType = openvinojs_common_1.jsTypeByPrecisionMap[precision];
    const heapTypeLabel = maps_js_1.heapLabelByArrayTypeMap[dataType.name];
    const originalDataPointer = originalTensor.getData();
    const elementsCount = shape.data.reduce((acc, val) => acc * val);
    const data = new dataType(elementsCount);
    for (let i = 0; i < elementsCount; i++) {
        // @ts-ignore: FIXME: Fix OpenvinoModule type
        const element = ov[heapTypeLabel][originalDataPointer / dataType.BYTES_PER_ELEMENT + i];
        data[i] = element;
    }
    return new openvinojs_common_1.Tensor(precision, data, shape);
}
exports.parseOriginalTensor = parseOriginalTensor;
function convertTensor(ov, tensor) {
    const { precision } = tensor;
    const dataType = openvinojs_common_1.jsTypeByPrecisionMap[precision];
    const originalShape = convertShape(ov, tensor.shape);
    const originalData = new dataType(tensor.data);
    const elementSizeInBytes = originalData.BYTES_PER_ELEMENT;
    const heapSpace = ov._malloc(originalData.length * elementSizeInBytes);
    const offset = Math.log2(elementSizeInBytes);
    const waPrecision = maps_js_1.heapLabelByArrayTypeMap[dataType.name];
    // @ts-ignore: FIXME: Fix OpenvinoModule type
    ov[waPrecision].set(originalData, heapSpace >> offset);
    return {
        obj: new ov.Tensor(precision, heapSpace, originalShape.obj),
        free: () => {
            originalShape.free();
            ov._free(heapSpace);
        }
    };
}
exports.convertTensor = convertTensor;
//# sourceMappingURL=helpers.js.map