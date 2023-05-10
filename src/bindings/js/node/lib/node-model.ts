import { Tensor, Shape, jsTypeByPrecisionMap } from 'openvinojs-common';

import type { ovNodeModule, NodeModel, NodeTensor } from './types';
import type {
  ITensor,
  IShape,
  IModel,
  PrecisionSupportedType,
} from 'openvinojs-common';

/* eslint-disable @typescript-eslint/no-var-requires */
const ovNode: ovNodeModule = require('../build/Release/ov_node_addon.node');

const DEFAULT_TENSOR_PRECISION = 'f32';

export default async function loadModel(xmlPath: string, binPath: string)
: Promise<IModel> {
  if (typeof xmlPath !== 'string' || typeof binPath !== 'string')
    throw new Error('Parameters \'xmlPath\' and \'binPath\' should be string');

  const model = new ovNode.Model().read_model(xmlPath).compile('CPU');

  return new CommonModel(ovNode, model);
}

function instanceOfITensor(object: any): object is ITensor {
  return 'data' in object;
}

class CommonModel implements IModel {
  #ovNode: ovNodeModule;
  #nodeModel: NodeModel;

  constructor(ovNode: ovNodeModule, nodeModel: NodeModel) {
    this.#ovNode = ovNode;
    this.#nodeModel = nodeModel;
  }

  async infer(tensorOrDataArray: ITensor | number[], shape: IShape)
  : Promise<ITensor> {
    const tensorData = instanceOfITensor(tensorOrDataArray)
      ? tensorOrDataArray.data
      : jsTypeByPrecisionMap[DEFAULT_TENSOR_PRECISION].from(tensorOrDataArray);

    const precision = tensorOrDataArray instanceof Tensor
      ? tensorOrDataArray.precision
      : DEFAULT_TENSOR_PRECISION;

    const nodeTensor = new this.#ovNode.Tensor(
      precision,
      shape.data,
      tensorData,
    );
    const output = this.#nodeModel.infer(nodeTensor);

    return parseNodeTensor(output);
  }
}

function parseNodeTensor(nodeTensor: NodeTensor): Tensor {
  const precision = nodeTensor.getPrecision() as PrecisionSupportedType;
  const data = nodeTensor.data;
  const shape = new Shape(nodeTensor.getShape().getData());

  return new Tensor(precision, data, shape);
}
