import Session from './session.mjs';

export default class SessionBrowser extends Session {
  constructor(ov) {
    super(ov, 'browser');
  }

  async loadModel(xmlPath, binPath, shape, layout) {
    const xmlData = await getFileDataAsArray(xmlPath);  
    const binData = await getFileDataAsArray(binPath);

    return super.loadModel(xmlData, binData, shape, layout);
  }
}

async function getFileDataAsArray(path) {
  const blob = await fetch(path).then(response => {
    if (!response.ok) {
      return null;
    }     
    return response.blob();
  });

  return new Uint8Array(await blob.arrayBuffer());
}
