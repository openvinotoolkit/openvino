--------------------------------------------------------------------------------
--
--  This file is part of the Doxyrest toolkit.
--
--  Doxyrest is distributed under the MIT license.
--  For details see accompanying license.txt file,
--  the public copy of which is also available at:
--  http://tibbo.com/downloads/archive/doxyrest/license.txt
--
--------------------------------------------------------------------------------

g_cfamilyUtilsIncluded = true

dofile(g_frameDir .. "/../common/string.lua")
dofile(g_frameDir .. "/../common/table.lua")
dofile(g_frameDir .. "/../common/item.lua")
dofile(g_frameDir .. "/../common/doc.lua")
dofile(g_frameDir .. "/../common/toc.lua")

if not LANGUAGE then
	LANGUAGE = "cpp"
end

if not INDEX_TITLE then
	INDEX_TITLE = "My Project Documentation"
end

if not EXTRA_PAGE_LIST then
	EXTRA_PAGE_LIST = {}
end

if not EXTERNAL_CREF_DB then
	EXTERNAL_CREF_DB = {}
end

if ML_PARAM_LIST_COUNT_THRESHOLD then
	ML_PARAM_LIST_COUNT_THRESHOLD = tonumber(ML_PARAM_LIST_COUNT_THRESHOLD)
end

if ML_PARAM_LIST_LENGTH_THRESHOLD then
	ML_PARAM_LIST_LENGTH_THRESHOLD = tonumber(ML_PARAM_LIST_LENGTH_THRESHOLD)
end

if string.match(LANGUAGE, "^c[px+]*$") then
	g_nameDelimiter = "::"
else
	g_nameDelimiter = "."
end

g_protectionKindMap = {
	public    = 0,
	protected = 1,
	private   = 2,
	package   = 3,
}

g_protectionNameMap = {
	[0] = "public",
	[1] = "protected",
	[2] = "private",
	[3] = "package",
}

g_minProtectionValue = 0
g_maxProtectionValue = 3

if PROTECTION_FILTER then
	g_protectionFilterValue = g_protectionKindMap[PROTECTION_FILTER]
else
	g_protectionFilterValue = 0
end

if PRE_PARAM_LIST_SPACE then
	g_preParamSpace = " "
	g_preOperatorParamSpace = ""
elseif PRE_OPERATOR_PARAM_LIST_SPACE then
	g_preParamSpace = ""
	g_preOperatorParamSpace = " "
else
	g_preParamSpace = ""
	g_preOperatorParamSpace = ""
end

if PRE_OPERATOR_NAME_SPACE then
	g_preOperatorNameSpace = " "
else
	g_preOperatorNameSpace = ""
end

if PRE_BODY_NL then
	g_preBodySpace = "\n\t"
else
	g_preBodySpace = " "
end

if not g_globalNamespace.title then
	g_globalNamespace.title = "Global Namespace"
end

g_currentCompoundId = nil

-------------------------------------------------------------------------------

function getNormalizedCppString(string)
	local s = string

	s = string.gsub(s, "%s*%*", "*")
	s = string.gsub(s, "%s*&", "&")
	s = string.gsub(s, "%s*<%s*", g_preParamSpace .. "<")
	s = string.gsub(s, "%s+>", ">")
	s = string.gsub(s, "%s*%(%s*", g_preParamSpace .. "(")
	s = string.gsub(s, "%s+%)", ")")

	return s
end

function getLinkedTextString(text, isRef)
	if not text then
		return ""
	end

	if not isRef then
		return getNormalizedCppString(text.plainText)
	end

	local s = ""

	for i = 1, #text.refTextArray do
		local refText = text.refTextArray[i]
		local text = getNormalizedCppString(refText.text)

		if refText.id ~= "" and refText.id ~= g_currentCompoundId then
			text = string.gsub(text, "<", "\\<") -- escape left chevron
			s = s .. ":ref:`" .. text .. "<doxid-" .. refText.id .. ">`"
		else
			s = s .. text
		end
	end

	s = string.gsub(s, "\n", " ") -- callsites of getLinkedTextString don't expect newline chars

	return s
end

-------------------------------------------------------------------------------

-- param array formatting

function getParamString(param, isRef)
	local s = ""
	local name

	if not param.type.isEmpty then
		s = s .. getLinkedTextString(param.type, isRef)
	end

	if param.declarationName ~= "" then
		name = param.declarationName
	else
		name = param.definitionName
	end

	if name ~= "" then
		if s ~= "" then
			s = s .. " "
		end

		s = s .. getNormalizedCppString(name)
	end

	if param.array ~= "" then
		s = s .. g_preParamSpace .. param.array
	end

	if not param.defaultValue.isEmpty then
		s = s .. " = " .. getLinkedTextString(param.defaultValue, isRef)
	end

	return s
end

function getParamArrayString_sl(paramArray, isRef, lbrace, rbrace)
	local s
	local count = #paramArray

	if count == 0 then
		s = lbrace .. rbrace
	else
		s = lbrace .. getParamString(paramArray[1], isRef)

		for i = 2, count do
			s = s .. ", " .. getParamString(paramArray[i], isRef)
		end

		s = s .. rbrace
	end

	return s
end

function getParamArrayString_ml(paramArray, isRef, lbrace, rbrace, indent, nl)
	local s
	local count = #paramArray

	if count == 0 then
		s = lbrace .. rbrace
	elseif count == 1  then
		s = lbrace .. getParamString(paramArray[1], isRef) .. rbrace
	else
		s = lbrace .. nl .. indent .. "\t"

		for i = 1, count do
			s = s .. getParamString(paramArray[i], isRef)

			if i ~= count then
				s = s .. ","
			end

			s = s .. nl .. indent .. "\t"
		end
		s = s .. rbrace
	end

	return s
end

function getParamArrayString(prefix, paramArray, isRef, lbrace, rbrace, indent, nl)
	if not lbrace then
		lbrace = "("
	end

	if not rbrace then
		rbrace = ")"
	end

	if not indent then
		indent = "\t"
	end

	if not nl then
		nl = "\n"
	end

	local s = ""
	local space = ""

	if PRE_PARAM_LIST_SPACE then
		space = " "
	end

	if ML_PARAM_LIST_COUNT_THRESHOLD and
		#paramArray > ML_PARAM_LIST_COUNT_THRESHOLD then
		s = getParamArrayString_ml(paramArray, isRef, lbrace, rbrace, indent, nl)
	else
		s = getParamArrayString_sl(paramArray, isRef, lbrace, rbrace)

		if ML_PARAM_LIST_LENGTH_THRESHOLD then
			local decl = prefix .. space .. s
			decl = replaceRolesWithPlainText(decl)

			if string.len(decl) > ML_PARAM_LIST_LENGTH_THRESHOLD then
				s = getParamArrayString_ml(paramArray, isRef, lbrace, rbrace, indent, nl)
			end
		end
	end

	s = space .. s
	return s
end

-------------------------------------------------------------------------------

function getItemKindString(item, itemKindString)
	local s = ""

	if item.modifiers ~= "" then
		s = item.modifiers .. " "
	end

	s = s .. itemKindString
	return s
end

function getItemImportArray(item)
	if item.importArray and next(item.importArray) ~= nil then
		return item.importArray
	end

	local text = getItemInternalDocumentation(item)
	local importArray = {}
	local i = 1
	for import in string.gmatch(text, ":import%(([^:]+)%)") do
		importArray[i] = import
		i = i + 1
	end

	return importArray
end

function getItemImportString(item)
	local importArray = getItemImportArray(item)
	if next(importArray) == nil then
		return ""
	end

	local indent = getLineText()
	local importPrefix
	local importSuffix

	if string.match(LANGUAGE, "^c[px+]*$") or LANGUAGE == "idl" then
		importPrefix = "#include <"
		importSuffix = ">\n" .. indent
	elseif string.match(LANGUAGE, "^ja?ncy?$") then
		importPrefix = "import \""
		importSuffix = "\"\n" .. indent
	else
		importPrefix = "import "
		importSuffix = "\n" .. indent
	end

	local s = ""

	for i = 1, #importArray do
		local import = importArray[i]
		s = s .. importPrefix .. import .. importSuffix
	end

	return s
end

function getEnumImportString(item)
	local importArray = getItemImportArray(item)
	if next(importArray) ~= nil then
		return getItemImportArray(item)
	end

	local file = string.gsub(item.location.file, ".*[\\/]", "")

	if string.match(LANGUAGE, "^c[px+]*$") and string.match(file, "%.h[px+]*$") then
		return "#include <" .. file .. ">\n"
	else
		return ""
	end
end

function getTemplateSpecSuffix(item)
	if not item.templateSpecParamArray or #item.templateSpecParamArray == 0 then
		return ""
	end

	return getParamArrayString_sl(item.templateSpecParamArray, false, "<", ">")
end

function getItemSimpleName(item)
	return item.name .. getTemplateSpecSuffix(item)
end

function getItemQualifiedName(item)
	local name = string.gsub(item.path, "/", g_nameDelimiter)
	return name .. getTemplateSpecSuffix(item)
end

getItemName = getItemQualifiedName

function getItemNameForOverview(item)
	local template = getItemNameTemplate(item)
	local name = getItemName(item)
	return fillItemNameTemplate(template, name, item.id)
end

function getFunctionModifierDelimiter(indent)
	if ML_SPECIFIER_MODIFIER_LIST then
		return "\n" .. indent
	else
		return " "
	end
end

function getPropertyDeclString(item, nameTemplate, indent)
	local s = getLinkedTextString(item.returnType, true)

	if item.modifiers ~= "" then
		s = string.gsub(s, "property$", item.modifiers .. " property")
	end

	s = s .. getFunctionModifierDelimiter(indent)
	s = s .. fillItemNameTemplate(nameTemplate, getItemName(item), item.id)

	if #item.paramArray > 0 then
		s = s .. getParamArrayString(s, item.paramArray, true, "(", ")", indent)
	end

	return s
end

function getFunctionDeclStringImpl(item, returnType, nameTemplate, indent)
	local s = ""

	if item.templateParamArray and #item.templateParamArray > 0 or
		item.templateSpecParamArray and #item.templateSpecParamArray > 0 then
		s = "template " .. getParamArrayString(s, item.templateParamArray, false, "<", ">") .. "\n" .. indent
	end

	if string.find(item.flags, "static") then
		s = s .. "static" .. getFunctionModifierDelimiter(indent)
	elseif item.virtualKind == "pure-virtual" then
		s = s .. "virtual" .. getFunctionModifierDelimiter(indent)
	elseif item.virtualKind ~= "non-virtual" then
		s = s .. item.virtualKind .. getFunctionModifierDelimiter(indent)
	end

	if returnType and returnType ~= "" then
		s = s .. returnType .. getFunctionModifierDelimiter(indent)
	end

	if item.modifiers ~= "" then
		s = s .. item.modifiers .. getFunctionModifierDelimiter(indent)
	end

	local name = getItemName(item)
	local isOperator = string.find(name, "operator")

	if isOperator then
		name = string.gsub(
			name,
			"operator%s*([()[%]+%-*&=!<>]+)%s*",
			"operator" .. g_preOperatorNameSpace .. "%1"
			)

		name = string.gsub(name, "%%", "%%%%")
	end

	s = s .. fillItemNameTemplate(nameTemplate, name, item.id)

	if isOperator then -- ensure spacce after operator
		s = s .. g_preOperatorParamSpace
	end

	s = s .. getParamArrayString(s, item.paramArray, true, "(", ")", indent)

	if string.find(item.flags, "const") then
		s = s .. " const"
	end

	if item.virtualKind == "pure-virtual" then
		s = s .. " = 0"
	end

	return s
end

function getFunctionDeclString(func, nameTemplate, indent)
	return getFunctionDeclStringImpl(
		func,
		getLinkedTextString(func.returnType, true),
		nameTemplate,
		indent
		)
end

function getVoidFunctionDeclString(func, nameTemplate, indent)
	return getFunctionDeclStringImpl(
		func,
		nil,
		nameTemplate,
		indent
		)
end

function getEventDeclString(event, nameTemplate, indent)
	return getFunctionDeclStringImpl(
		event,
		"event",
		nameTemplate,
		indent
		)
end

function getDefineDeclString(define, nameTemplate, indent)
	local s = "#define "

	s = s .. fillItemNameTemplate(nameTemplate, define.name, define.id)

	if #define.paramArray > 0 then
		-- no space between name and params!

		s = s .. getParamArrayString(s, define.paramArray, false, "(", ")", indent, " \\\n")
	end

	return s
end

function getTypedefDeclString(typedef, nameTemplate, indent)
	local s
	local name = fillItemNameTemplate(nameTemplate, getItemName(typedef), typedef.id)
	local declName

	if TYPEDEF_TO_USING then
		s = "using " .. name .. " ="
		declName = ""
	else
		s = "typedef"
		declName = name
	end

	if next(typedef.paramArray) == nil then
		local type = getLinkedTextString(typedef.type, true)
		if string.match(type, "%(%*$") then -- quickfix for function pointers
			s = s .. " " .. string.sub(type, 1, -3) ..  " (*"
		else
			s = s .. " " .. type .. " "
		end

		s = s .. declName

		if typedef.argString ~= "" then
			s = s .. formatArgDeclString(typedef.argString, indent)
		end

		return s
	end

 	s = s .. getFunctionModifierDelimiter(indent) .. getLinkedTextString(typedef.type, true) .. getFunctionModifierDelimiter(indent)
	s = s .. declName
	s = s .. getParamArrayString(s, typedef.paramArray, true, "(", ")", indent)

	return s
end

function getClassDeclString(class, nameTemplate, indent)
	local s = ""

	if #class.templateParamArray > 0 or #class.templateSpecParamArray > 0 then
		s = "template"
		if not PRE_PARAM_LIST_SPACE then
			s = s .. " "
		end
		s = s .. getParamArrayString(s, class.templateParamArray, false, "<", ">", indent) .. "\n" .. indent
	end

	s = s .. class.compoundKind .. " "
	s = s .. fillItemNameTemplate(nameTemplate, getItemName(class), class.id)

	return s
end

function getBaseClassString(class, protection)
	local s = ""

	if string.match(LANGUAGE, "^c[px+]*$") then
		s = protection .. " "
	end

	if class.id ~= "" then
		s = s .. ":ref:`" .. getItemQualifiedName(class) .. "<doxid-" .. class.id .. ">`"
	else
		-- class without id (imported)

		local url = EXTERNAL_CREF_DB[class.name]
		if url then
			s = s .. ":ref:`" .. class.name .. "<" .. url .. ">`"
		else
			s = s .. class.name
		end
	end

	return s
end

function isMemberOfUnnamedType(item)
	local text = getItemInternalDocumentation(item)
	return string.match(text, ":unnamed%(([%w/:]+)%)")
end

g_closingBracketChar =
{
	["("] = ")",
	["["] = "]",
	["<"] = ">",
	["{"] = "}",
}

function formatArgDeclString(decl, indent)
	local bracket = {}
	bracket[0] = {}
	bracket[0].result = ""

	local level = 0
	local pos = 1
	local len = string.len(decl)

	for i = 1, len do
		local c = string.sub(decl, i, i)

		if c == "(" or c == "[" or c == "<" or c == "{" then
			local chunk = trimLeadingWhitespace(string.sub(decl, pos, i))
			pos = i + 1

			bracket[level].result = bracket[level].result .. chunk

			level = level + 1
			bracket[level] = {}
			bracket[level].closingChar = g_closingBracketChar[c]
			bracket[level].result = ""
		elseif level > 0 then
			if c == ',' then
				if not bracket[level].delimiter then
					bracket[level].delimiter = "\n" .. indent .. string.rep("\t", level)
					bracket[level].result = bracket[level].delimiter .. bracket[level].result
				end

				local chunk = trimLeadingWhitespace(string.sub(decl, pos, i))
				pos = i + 1

				bracket[level].result = bracket[level].result .. chunk .. bracket[level].delimiter
				pos = i + 1
			elseif c == bracket[level].closingChar then
				local chunk = trimLeadingWhitespace(string.sub(decl, pos, i - 1))
				pos = i

				bracket[level].result = bracket[level].result .. chunk

				if bracket[level].delimiter then
					bracket[level].result = bracket[level].result .. bracket[level].delimiter
				end

				level = level - 1
				bracket[level].result = bracket[level].result .. bracket[level + 1].result
			end
		end
	end

	if pos <= len then
		bracket[0].result = bracket[0].result .. string.sub(decl, pos)
	end

	return bracket[0].result
end

function getNamespaceTree(nspace, indent)
	local s = ""

	if not indent then
		indent = ""
	end

	s = "\t" .. indent .. "namespace :ref:`" .. getItemQualifiedName(nspace) .. "<doxid-" .. nspace.id ..">`;\n"

	for i = 1, #nspace.namespaceArray do
		s = s .. getNamespaceTree(nspace.namespaceArray[i], indent .. "\t")
	end

	return s
end

-------------------------------------------------------------------------------

-- item filtering utils

function getDefaultProtectionKind(compound)
	if compound.compoundKind ~= "class" or string.match(LANGUAGE, "^ja?ncy?$") then
		return 0
	else
		return 2
	end
end

function getProtectionValue(protectionKind)
	local protectionValue = g_protectionKindMap[protectionKind]

	if not protectionValue then
		protectionValue = 0 -- assume public
	end

	return protectionValue
end

function isItemExcludedByProtectionFilter(item)

	local protectionValue = g_protectionKindMap[item.protectionKind]
	if protectionValue and protectionValue > g_protectionFilterValue then
		return true
	end

	return false
end

function hasNonPublicItems(array)
	if next(array) == nil then
		return false
	end

	local lastItem = array[#array]
	local protectionValue = g_protectionKindMap[lastItem.protectionKind]
	return protectionValue and protectionValue > 0
end

function isItemExcludedByLocationFilter(item)
	if not item.location then
		return false
	end

	return
		-- exclude explicitly specified locations
		EXCLUDE_LOCATION_PATTERN and
		string.match(item.location.file, EXCLUDE_LOCATION_PATTERN)
end

function filterItemArray(itemArray)
	if next(itemArray) == nil then
		return
	end

	for i = #itemArray, 1, -1 do
		local item = itemArray[i]
		local isExcluded =
			isItemExcludedByProtectionFilter(item) or
			isItemExcludedByLocationFilter(item)

		if isExcluded then
			table.remove(itemArray, i)
		end
	end
end

function filterEnumArray(enumArray)
	if next(enumArray) == nil then
		return
	end

	for i = #enumArray, 1, -1 do
		local enum = enumArray[i]
		local isExcluded =
			isItemExcludedByProtectionFilter(enum) or
			isItemExcludedByLocationFilter(enum) or
			isUnnamedItem(enum) and #enum.enumValueArray == 0

		if isExcluded then
			table.remove(enumArray, i)
		end
	end
end

function filterNamespaceArray(namespaceArray)
	if next(namespaceArray) == nil then
		return
	end

	for i = #namespaceArray, 1, -1 do
		local item = namespaceArray[i]
		local isExcluded =
			isUnnamedItem(item) or
			isItemExcludedByLocationFilter(item)

		if isExcluded then
			table.remove(namespaceArray, i)
		end
	end
end

function filterConstructorArray(constructorArray)
	filterItemArray(constructorArray)

	if #constructorArray == 1 then
		local item = constructorArray[1]
		local isExcluded =
			isItemExcludedByProtectionFilter(item) or
			EXCLUDE_DEFAULT_CONSTRUCTORS and #item.paramArray == 0

		if isExcluded then
			table.remove(constructorArray, 1)
		end
	end
end

function filterDefineArray(defineArray)
	if next(defineArray) == nil then
		return
	end

	for i = #defineArray, 1, -1 do
		local item = defineArray[i]

		local isExcluded =
			isItemExcludedByLocationFilter(item) or
			EXCLUDE_EMPTY_DEFINES and item.initializer.isEmpty or
			EXCLUDE_DEFINE_PATTERN and string.match(item.name, EXCLUDE_DEFINE_PATTERN)

		if isExcluded then
			table.remove(defineArray, i)
		end
	end
end

function filterTypedefArray(typedefArray)
	if next(typedefArray) == nil then
		return
	end

	for i = #typedefArray, 1, -1 do
		local item = typedefArray[i]
		local isExcluded =
			isItemExcludedByProtectionFilter(item) or
			isItemExcludedByLocationFilter(item)

		if not isExcluded and EXCLUDE_PRIMITIVE_TYPEDEFS then
			local typeKind, name = string.match(
				item.type.plainText,
				"(%a+)%s+(%w[%w_]*)"
				)

			if name == item.name then
				isExcluded =
					item.briefDescription.isEmpty and
					item.detailedDescription.isEmpty
			end
		end

		if isExcluded then
			table.remove(typedefArray, i)
		end
	end
end

-------------------------------------------------------------------------------

-- base compound is an artificial compound holding all inherited members

function addToBaseCompound(baseCompound, baseTypeArray)
	for i = 1, #baseTypeArray do
		local baseType = baseTypeArray[i]

		-- prevent adding the same base type multiple times

		if not baseCompound.baseTypeMap[baseType] and
			baseType.compoundKind ~= "<undefined>" then

			baseCompound.baseTypeMap[baseType] = true
			prepareCompound(baseType)

			if next(baseType.baseTypeArray) ~= nil then
				addToBaseCompound(baseCompound, baseType.baseTypeArray)
			end

			concatenateTables(baseCompound.typedefArray, baseType.typedefArray)
			concatenateTables(baseCompound.enumArray, baseType.enumArray)
			concatenateTables(baseCompound.structArray, baseType.structArray)
			concatenateTables(baseCompound.unionArray, baseType.unionArray)
			concatenateTables(baseCompound.interfaceArray, baseType.interfaceArray)
			concatenateTables(baseCompound.protocolArray, baseType.protocolArray)
			concatenateTables(baseCompound.exceptionArray, baseType.exceptionArray)
			concatenateTables(baseCompound.classArray, baseType.classArray)
			concatenateTables(baseCompound.singletonArray, baseType.singletonArray)
			concatenateTables(baseCompound.serviceArray, baseType.serviceArray)
			concatenateTables(baseCompound.variableArray, baseType.variableArray)
			concatenateTables(baseCompound.propertyArray, baseType.propertyArray)
			concatenateTables(baseCompound.eventArray, baseType.eventArray)
			concatenateTables(baseCompound.functionArray, baseType.functionArray)
			concatenateTables(baseCompound.aliasArray, baseType.aliasArray)
		end
	end
end

function createPseudoCompound(compoundKind)
	local compound = {}

	compound.compoundKind = compoundKind
	compound.baseTypeMap = {}
	compound.namespaceArray = {}
	compound.typedefArray = {}
	compound.enumArray = {}
	compound.structArray = {}
	compound.unionArray = {}
	compound.interfaceArray = {}
	compound.protocolArray = {}
	compound.exceptionArray = {}
	compound.classArray = {}
	compound.singletonArray = {}
	compound.serviceArray = {}
	compound.variableArray = {}
	compound.propertyArray = {}
	compound.eventArray = {}
	compound.constructorArray = {}
	compound.functionArray = {}
	compound.aliasArray = {}
	compound.defineArray = {}

	return compound
end

function createBaseCompound(compound)
	local baseCompound = createPseudoCompound("base-compound")
	baseCompound.isBaseCompound = true
	addToBaseCompound(baseCompound, compound.baseTypeArray)
	baseCompound.hasItems = hasCompoundItems(baseCompound)
	handleCompoundProtection(baseCompound)

	compound.baseCompound = baseCompound
end

function sortByProtection(array)
	if next(array) == nil then
		return
	end

	local bucketArray  = {}
	for i = g_minProtectionValue, g_maxProtectionValue do
		bucketArray[i] = {}
	end

	for i = 1, #array do
		local item = array[i]
		local protectionValue = getProtectionValue(item.protectionKind)
		table.insert(bucketArray[protectionValue], item)
	end

	local result = {}
	local k = 1

	for i = g_minProtectionValue, g_maxProtectionValue do
		local bucket = bucketArray[i]

		for j = 1, #bucket do
			array[k] = bucket[j]
			k = k + 1
		end
	end

	assert(k == #array + 1)

	return #array - #bucketArray[0]
end

function addToProtectionCompounds(compound, arrayName)
	local array = compound[arrayName]

	if next(array) == nil then
		return
	end

	for i = 1, #array do
		local item = array[i]
		local protectionValue = getProtectionValue(item.protectionKind)
		local protectionCompound = compound.protectionCompoundArray[protectionValue]
		local targetArray = protectionCompound[arrayName]

		table.insert(targetArray, item)
		protectionCompound.isEmpty = false
	end
end

function handleCompoundProtection(compound)

	compound.protectionCompoundArray = {}

	for i = g_minProtectionValue, g_maxProtectionValue do
		local protectionCompound = createPseudoCompound("protection-compound")
		protectionCompound.isEmpty = true
		protectionCompound.isBaseCompound = compound.isBaseCompound
		compound.protectionCompoundArray[i] = protectionCompound
	end

	addToProtectionCompounds(compound, "namespaceArray")
	addToProtectionCompounds(compound, "typedefArray")
	addToProtectionCompounds(compound, "enumArray")
	addToProtectionCompounds(compound, "structArray")
	addToProtectionCompounds(compound, "unionArray")
	addToProtectionCompounds(compound, "interfaceArray")
	addToProtectionCompounds(compound, "protocolArray")
	addToProtectionCompounds(compound, "exceptionArray")
	addToProtectionCompounds(compound, "classArray")
	addToProtectionCompounds(compound, "singletonArray")
	addToProtectionCompounds(compound, "serviceArray")
	addToProtectionCompounds(compound, "variableArray")
	addToProtectionCompounds(compound, "propertyArray")
	addToProtectionCompounds(compound, "eventArray")
	addToProtectionCompounds(compound, "constructorArray")
	addToProtectionCompounds(compound, "functionArray")
	addToProtectionCompounds(compound, "aliasArray")
	addToProtectionCompounds(compound, "defineArray")

	if compound.destructor then
		local protectionValue = getProtectionValue(compound.destructor.protectionKind)
		local protectionCompound = compound.protectionCompoundArray[protectionValue]
		protectionCompound.destructor = compound.destructor
		protectionCompound.isEmpty = false
	end

	-- eliminate empty protection compounds

	for i = g_minProtectionValue, g_maxProtectionValue do
		if compound.protectionCompoundArray[i].isEmpty then
			compound.protectionCompoundArray[i] = nil
		end
	end
end

-------------------------------------------------------------------------------

-- compound & enum prep

function hasCompoundItems(compound)
	return
		#compound.namespaceArray ~= 0 or
		#compound.typedefArray ~= 0 or
		#compound.enumArray ~= 0 or
		#compound.structArray ~= 0 or
		#compound.unionArray ~= 0 or
		#compound.interfaceArray ~= 0 or
		#compound.protocolArray ~= 0 or
		#compound.exceptionArray ~= 0 or
		#compound.classArray ~= 0 or
		#compound.singletonArray ~= 0 or
		#compound.serviceArray ~= 0 or
		#compound.variableArray ~= 0 or
		#compound.propertyArray ~= 0 or
		#compound.eventArray ~= 0 or
		#compound.constructorArray ~= 0 or
		#compound.functionArray ~= 0 or
		#compound.aliasArray ~= 0 or
		#compound.defineArray ~= 0
end

function prepareCompound(compound)
	if compound.stats then
		return compound.stats
	end

	local stats = {}

	-- filter invisible items out

	if compound.derivedTypeArray then
		filterItemArray(compound.derivedTypeArray)
	end

	filterNamespaceArray(compound.namespaceArray)
	filterTypedefArray(compound.typedefArray)
	filterEnumArray(compound.enumArray)
	filterItemArray(compound.structArray)
	filterItemArray(compound.unionArray)
	filterItemArray(compound.interfaceArray)
	filterItemArray(compound.protocolArray)
	filterItemArray(compound.exceptionArray)
	filterItemArray(compound.classArray)
	filterItemArray(compound.singletonArray)
	filterItemArray(compound.serviceArray)
	filterItemArray(compound.variableArray)
	filterItemArray(compound.propertyArray)
	filterItemArray(compound.eventArray)
	filterConstructorArray(compound.constructorArray)
	filterItemArray(compound.functionArray)
	filterItemArray(compound.aliasArray)
	filterDefineArray(compound.defineArray)

	stats.hasItems = hasCompoundItems(compound)

	-- scan for documentation and create subgroups

	stats.hasUnnamedEnums = false
	stats.hasDocumentedUnnamedEnumValues = false

	for i = 1, #compound.enumArray do
		local item = compound.enumArray[i]

		if isUnnamedItem(item) then
			stats.hasUnnamedEnums = true

			if prepareItemArrayDocumentation(item.enumValueArray, compound) then
				stats.hasDocumentedUnnamedEnumValues = true
			end
		end
	end

	stats.hasDocumentedTypedefs = prepareItemArrayDocumentation(compound.typedefArray, compound)
	stats.hasDocumentedVariables = prepareItemArrayDocumentation(compound.variableArray, compound)
	stats.hasDocumentedProperties = prepareItemArrayDocumentation(compound.propertyArray, compound)
	stats.hasDocumentedEvents = prepareItemArrayDocumentation(compound.eventArray, compound)
	stats.hasDocumentedFunctions = prepareItemArrayDocumentation(compound.functionArray, compound)
	stats.hasDocumentedAliases = prepareItemArrayDocumentation(compound.aliasArray, compound)
	stats.hasDocumentedDefines = prepareItemArrayDocumentation(compound.defineArray, compound)
	stats.hasDocumentedConstruction = prepareItemArrayDocumentation(compound.constructorArray, compound)

	if not EXCLUDE_DESTRUCTORS and compound.destructor then
		prepareItemDocumentation(compound.destructor, compound)
		stats.hasDocumentedConstruction = stats.hasDocumentedConstruction or compound.destructor.hasDocumentation
	end

	stats.hasDocumentedItems =
		stats.hasDocumentedUnnamedEnumValues or
		stats.hasDocumentedTypedefs or
		stats.hasDocumentedVariables or
		stats.hasDocumentedProperties or
		stats.hasDocumentedEvents or
		stats.hasDocumentedConstruction or
		stats.hasDocumentedFunctions or
		stats.hasDocumentedAliases or
		stats.hasDocumentedDefines

	stats.hasBriefDocumentation = not isDocumentationEmpty(compound.briefDescription)
	stats.hasDetailedDocumentation = not isDocumentationEmpty(compound.detailedDescription)

	-- sort items -- only the ones producing separate pages;
	-- also, defines, which always go to the global namespace

	table.sort(compound.groupArray, cmpGroups)
	table.sort(compound.namespaceArray, cmpNames)
	table.sort(compound.enumArray, cmpNames)
	table.sort(compound.structArray, cmpNames)
	table.sort(compound.unionArray, cmpNames)
	table.sort(compound.interfaceArray, cmpNames)
	table.sort(compound.protocolArray, cmpNames)
	table.sort(compound.exceptionArray, cmpNames)
	table.sort(compound.classArray, cmpNames)
	table.sort(compound.serviceArray, cmpNames)
	table.sort(compound.defineArray, cmpNames)

	compound.stats = stats

	if compound.baseTypeArray and next(compound.baseTypeArray) ~= nil then
		createBaseCompound(compound)
	end

	handleCompoundProtection(compound)

	return stats
end

function prepareEnum(enum)
	local stats = {}

	local maxLength = 0
	for i = 1, #enum.enumValueArray do
		local item = enum.enumValueArray[i]
		local length = string.len(item.name)
		if length > maxLength then
			maxLength = length
		end
	end

	stats.hasDocumentedEnumValues = prepareItemArrayDocumentation(enum.enumValueArray)
	stats.hasBriefDocumentation = not isDocumentationEmpty(enum.briefDescription)
	stats.hasDetailedDocumentation = not isDocumentationEmpty(enum.detailedDescription)
	stats.maxEnumValueNameLength = maxLength

	return stats
end

-------------------------------------------------------------------------------
