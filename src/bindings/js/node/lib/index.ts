const ov = require('../build/Release/ov_node_addon.node');
import loadModel from './node-model';
import createModule from 'openvinojs-common'


export default
  createModule('node', loadModel, getVersionString, getDescriptionString);


async function getVersionString(): Promise<string> {
    const str= "Version";
    return str;
};

async function getDescriptionString(): Promise<string> {
    return ov.getDescriptionString();
};



