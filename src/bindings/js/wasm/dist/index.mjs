// if (global) {
//   console.log('== it is nodejs');
// }
// else {
//   console.log('== it is browser');
// }
if (isNodeEnv()) {
    console.log('== is Node');
}
else {
    console.log('== is browser');
}
function isNodeEnv() {
    return import.meta.url.startsWith('file:');
}
export {};
//# sourceMappingURL=index.mjs.map