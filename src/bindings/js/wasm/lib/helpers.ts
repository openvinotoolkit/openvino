import {
  Shape,
  Tensor,
  IShape,
  ITensor,
  jsTypeByPrecisionMap,
} from 'openvinojs-common';

import { heapLabelByArrayTypeMap } from './maps';

import type { TypedArray } from 'openvinojs-common';
import type {
  HEAPType,
  OriginalShape,
  OriginalTensor,
  OpenvinoWASMModule,
  OriginalShapeWrapper,
  OriginalTensorWrapper,
} from './types';

export function isNodeEnv() {
  return typeof window === 'undefined';
}

export async function getFileDataAsArray(path: string): Promise<Uint8Array> {
  const fileData = isNodeEnv()
    ? await getFileDataNode(path)
    : await getFileDataBrowser(path);

  if (!fileData) throw new Error(`File '${path}' cannot be loaded!`);

  return new Uint8Array(fileData);
}

async function getFileDataNode(path: string): Promise<Buffer> {
  const { readFileSync } = await import('fs');

  return readFileSync(path);
}

async function getFileDataBrowser(path: string): Promise<ArrayBuffer | null> {
  const blob = await fetch(path).then(
    response => !response.ok ? null : response.blob()
  );

  return blob ? await blob.arrayBuffer() : null;
}

export function uploadFile(
  ov: OpenvinoWASMModule,
  filename: string,
  data: Uint8Array
) {
  const stream = ov.FS.open(filename, 'w+');

  ov.FS.write(stream, data, 0, data.length, 0);
  ov.FS.close(stream);
}

const SHAPE_HEAP: HEAPType = heapLabelByArrayTypeMap[Shape.TYPE.name];

export function parseOriginalShape(
  ov: OpenvinoWASMModule,
  originalShape: OriginalShape
): Shape {
  const originalDim = originalShape.getDim();
  const originalDataPointer = originalShape.getData();

  const dimensions = new Shape.TYPE(originalDim);

  for (let i = 0; i < originalDim; i++) {
    const dimension =
      ov[SHAPE_HEAP][originalDataPointer / Shape.TYPE.BYTES_PER_ELEMENT + i];

    dimensions[i] = dimension;
  }

  return new Shape(...dimensions);
}

export function convertShape(
  ov: OpenvinoWASMModule,
  shape: IShape
): OriginalShapeWrapper {
  const originalDimensions = new Shape.TYPE(shape.data);
  const elementSizeInBytes = originalDimensions.BYTES_PER_ELEMENT;
  const heapSpace = ov._malloc(originalDimensions.length*elementSizeInBytes);
  const offset = Math.sqrt(elementSizeInBytes);
  ov[SHAPE_HEAP].set(originalDimensions, heapSpace>>offset);

  return {
    obj: new ov.Shape(heapSpace, shape.dim),
    free: () => ov._free(heapSpace),
  };
}

export function parseOriginalTensor(
  ov: OpenvinoWASMModule,
  originalTensor: OriginalTensor
): Tensor {
  const precision = originalTensor.getPrecision();
  const shape = parseOriginalShape(ov, originalTensor.getShape());

  const dataType = jsTypeByPrecisionMap[precision];
  const heapTypeLabel = heapLabelByArrayTypeMap[dataType.name];
  const originalDataPointer = originalTensor.getData();

  const elementsCount =
    shape.data.reduce((acc: number, val: number) => acc*val);
  const data: TypedArray = new dataType(elementsCount);

  for (let i = 0; i < elementsCount; i++)
    data[i] =
      ov[heapTypeLabel][originalDataPointer/dataType.BYTES_PER_ELEMENT + i];

  return new Tensor(precision, data, shape);
}

export function convertTensor(
  ov: OpenvinoWASMModule,
  tensor: ITensor
): OriginalTensorWrapper {
  const { precision, data } = tensor;
  const originalShape = convertShape(ov, tensor.shape);
  const dataType = jsTypeByPrecisionMap[precision];
  const elementSizeInBytes = data.BYTES_PER_ELEMENT;
  const heapSpace = ov._malloc(data.length*elementSizeInBytes);
  const offset = Math.log2(elementSizeInBytes);
  const waPrecision = heapLabelByArrayTypeMap[dataType.name];

  ov[waPrecision].set(data, heapSpace>>offset);

  return {
    obj: new ov.Tensor(precision, heapSpace, originalShape.obj),
    free: () => {
      originalShape.free();
      ov._free(heapSpace);
    },
  };
}
