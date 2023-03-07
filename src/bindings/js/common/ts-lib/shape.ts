import { 
  JSArrayType, 
  HEAPType, 
  PrecisionSupportedTypes,
  IShape,
} from './types';
import { jsTypeByPrecisionMap, heapLabelByArrayTypeMap } from './maps';

import { OpenvinoModule, OriginalShape, OriginalShapeWrapper } from './ov-module';

export default class Shape implements IShape {
  #dimensions: Uint32Array;

  static TYPE: JSArrayType = jsTypeByPrecisionMap[PrecisionSupportedTypes.uint32];
  static HEAP: HEAPType = heapLabelByArrayTypeMap[Shape.TYPE.name];

  constructor(dimensionsArray: number[]);
  constructor(...dimensionsArray: number[]);
  constructor(...dimensionsArray: number[] | [number[]]) {
    const dimensions: number[] = Array.isArray(dimensionsArray[0]) ? dimensionsArray[0] : dimensionsArray as number[];

    this.#dimensions = new Uint32Array(dimensions.length);

    dimensions.map((dimension, index): void => { this.#dimensions[index] = dimension });
  }

  get dim(): number {
    return this.#dimensions.length;
  }

  get data(): Uint32Array {
    return this.#dimensions;
  }
  
  static parse(ov: OpenvinoModule, originalShape: OriginalShape): Shape {
    const originalDim = originalShape.getDim();
    const originalDataPointer = originalShape.getData();

    const dimensions = new Shape.TYPE(originalDim);

    for (let i = 0; i < originalDim; i++) {
      const dimension = ov[Shape.HEAP][originalDataPointer/Shape.TYPE.BYTES_PER_ELEMENT + i];
      
      dimensions[i] = dimension;
    }

    return new Shape(...dimensions);
  }

  static convert(ov: OpenvinoModule, shape: Shape): OriginalShapeWrapper {
    const originalDimensions = new Shape.TYPE(shape.data);
    const elementSizeInBytes = originalDimensions.BYTES_PER_ELEMENT;
    const heapSpace = ov._malloc(originalDimensions.length*elementSizeInBytes);
    const offset = Math.sqrt(elementSizeInBytes);
    ov[Shape.HEAP].set(originalDimensions, heapSpace>>offset); 
    
    return { obj: new ov.Shape(heapSpace, shape.dim), free: () => ov._free(heapSpace) };
  }

  convert(ov: OpenvinoModule): OriginalShapeWrapper {
    return Shape.convert(ov, this);
  }

  toString(): string {
    return `[${this.#dimensions.join(',')}]`;
  }
}
