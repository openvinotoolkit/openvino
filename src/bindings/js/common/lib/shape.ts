import { PrecisionSupportedTypes } from './types.js';
import { jsTypeByPrecisionMap } from './maps.js';

import type { JSArrayType, IShape,
} from './types.js';

export default class Shape implements IShape {
  #dimensions: Uint32Array;

  static TYPE: JSArrayType = jsTypeByPrecisionMap[PrecisionSupportedTypes.uint32];

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

  toString(): string {
    return `[${this.#dimensions.join(',')}]`;
  }
}
