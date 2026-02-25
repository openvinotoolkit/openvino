const { defineConfig } = require("eslint/config");
const { configs } = require("@eslint/js");
const globals = require("globals");

module.exports = defineConfig({
    ignores: ["node_modules/"],
    extends: [configs.recommended],
    languageOptions: {
        globals: globals.node,
        parserOptions: {
            ecmaVersion: "latest",
        },
    },
});
