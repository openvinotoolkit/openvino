const eslint = require("@eslint/js");
const prettierConfig = require("eslint-config-prettier/flat");
const globals = require("globals");
const tseslint = require("typescript-eslint");
const { defineConfig } = require("eslint/config");

module.exports = defineConfig([
  eslint.configs.recommended,
  tseslint.configs.recommendedTypeChecked,
  prettierConfig, // to disable stylistic rules from ESLint
  {
    ignores: ["types/", "dist/"],
  },
  {
    files: ["**/*.{js,mjs,cjs,ts}"],
    languageOptions: {
      globals: globals.node,
      parser: tseslint.parser,
    },
    rules: {
      "no-var": ["error"],
      camelcase: ["error"],
      "prefer-destructuring": ["error", { object: true, array: false }],
      "@typescript-eslint/no-explicit-any": 0,
      "@typescript-eslint/no-require-imports": 0,
    },
  },
  {
    files: ["**/addon.ts"],
    rules: {
      "@typescript-eslint/no-misused-new": "off",
    },
  },
]);
