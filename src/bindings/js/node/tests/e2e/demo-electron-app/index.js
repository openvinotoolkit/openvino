const { app } = require('electron');
const { addon: ov } = require('openvino-node');

app.whenReady().then(() => {
  console.log('Creating OpenVINO Runtime Core');
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const core = new ov.Core();
  console.log('Created OpenVINO Runtime Core');

  app.exit(0);
});
