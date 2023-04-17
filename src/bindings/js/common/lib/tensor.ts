import { jsTypeByPrecisionMap } from './maps.js';

import Shape from './shape.js';

import type { 
  TypedArray,
  PrecisionSupportedType, 
  IShape,
  ITensor,
} from './types.js';

export default class Tensor implements ITensor {
  #precision: PrecisionSupportedType;
  #data: TypedArray;
  #shape: IShape;

  constructor(precision: PrecisionSupportedType, data: number[] | TypedArray, shapeData: IShape | number[]) {
    this.#precision = precision;
    this.#data = new jsTypeByPrecisionMap[this.#precision](data);
    
    if (shapeData instanceof Shape) this.#shape = shapeData;
    else this.#shape = new Shape(...shapeData as number[]);
  }

  get precision(): PrecisionSupportedType {
    return this.#precision;
  }

  get data(): TypedArray {
    return this.#data;
  }

  get shape(): IShape {
    return this.#shape;
  }
};
