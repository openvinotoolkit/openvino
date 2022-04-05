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

g_itemFileNameMap = {}

function ensureUniqueItemName(item, name, map, sep)
	local mapValue = map[name]

	if mapValue == nil then
		mapValue = {}
		mapValue.itemMap = {}
		mapValue.itemMap[item.id] = 1
		mapValue.items = {}
		mapValue.count = 1
		mapValue.items[mapValue.count] = item
		map[name] = mapValue
	else
		local index = mapValue.itemMap[item.id]

		if index == nil then
			index = mapValue.count + 1
			mapValue.itemMap[item.id] = index
			mapValue.items[index] = item
			mapValue.count = mapValue.count + 1
		end
	end

	return name
end

function getItemFileName(item, suffix)
	local s
	
	if item.compoundKind == 'page' then
		s = ''
	elseif item.compoundKind == "interface" then
		s = 'class'
	elseif item.compoundKind then
		s = item.compoundKind
	elseif item.memberKind then
		s = item.memberKind
	else
		s = "undef_"
	end

	if item.compoundKind == "group" then
		s = s .. item.name
	else
		local path = string.gsub(item.path, "/operator[%s%p]+$", "/operator")
		s = s .. string.gsub(path, "/", "_1_1")
	end

	s = ensureUniqueItemName(item, s, g_itemFileNameMap, "-")

	if not suffix then
		suffix = ".rst"
	end

	return s .. suffix
end

function getItemFileNameNoSuffix(item)
	local s
	
	if item.compoundKind == 'page' then
		s = ''
	elseif item.compoundKind == "interface" then
		s = 'class'
	elseif item.compoundKind then
		s = item.compoundKind
	elseif item.memberKind then
		s = item.memberKind
	else
		s = "undef_"
	end

	if item.compoundKind == "group" then
		s = s .. item.name
	else
		local path = string.gsub(item.path, "/operator[%s%p]+$", "/operator")
		s = s .. string.gsub(path, "/", "_1_1")
	end

	return s
end

function getItemRefTargetString(item)
	if string.match(item.name, ":$") then -- Objective C methods (simple handling for now)
		return ".. _doxid-" .. item.id .. ":\n"
	end

	local s =
		".. _doxid-" .. item.id .. ":\n" ..
		".. index:: pair: " .. item.memberKind .. "; " .. item.name .. "\n"

	if item.isSubGroupHead then
		for j = 1, #item.subGroupSlaveArray do
			slaveItem = item.subGroupSlaveArray[j]

			s = s ..
				".. _doxid-" .. slaveItem.id .. ":\n" ..
				".. index:: pair: " .. slaveItem.memberKind .. "; " .. slaveItem.name .. "\n"
		end
	end

	return s
end

function hasItemDocumentation(item)
	return item.hasDocumentation or item.subGroupHead -- in which case subgroup head has doc
end

g_simpleItemNameTemplate = "$n"
g_refItemNameTemplate = ":ref:`$n<doxid-$i>`"
g_targetItemNameTemplate = ":target:`$n<doxid-$i>`"

function getItemNameTemplate(item)
	if hasItemDocumentation(item) then
		return g_refItemNameTemplate
	else
		return g_targetItemNameTemplate
	end
end

function fillItemNameTemplate(template, name, id)
	local s = string.gsub(template, "$n", name)
	s = string.gsub(s, "$i", id)
	return s
end

function isTocTreeItem(compound, item)
	return not item.group or item.group == compound
end

function isUnnamedItem(item)
	return item.name == "" or string.sub(item.name, 1, 1) == "@"
end

function getItemInternalDocumentation(item)
	local s = ""

	for i = 1, #item.detailedDescription.docBlockList do
		local block = item.detailedDescription.docBlockList[i]
		if block.blockKind == "internal" then
			s = s .. block.text .. getSimpleDocBlockListContents(block.childBlockList)
		end
	end

	return s
end

function getItemBriefDocumentation(item)
	local s = getDocBlockListContents(item.briefDescription.docBlockList)

	if string.len(s) == 0 then
		s = getDocBlockListContents(item.detailedDescription.docBlockList)
		if string.len(s) == 0 then
			return ""
		end

		-- generate brief description from first sentence only
		-- matching space is to handle qualified identifiers(e.g. io.File.open)

		local i = string.find(s, "%.%s", 1)
		if i then
			s = string.sub(s, 1, i)
		end

		s = trimTrailingWhitespace(s)
	end

	return s .. " :ref:`More...<details-" .. item.id .. ">`"
end

function getItemDetailedDocumentation(item)
	local brief = getDocBlockListContents(item.briefDescription.docBlockList)
	local detailed = getDocBlockListContents(item.detailedDescription.docBlockList)

	if string.len(detailed) == 0 then
		return brief
	elseif string.len(brief) == 0 then
		return detailed
	else
		return brief .. "\n\n" .. detailed
	end
end

function prepareItemDocumentation(item, compound)
	local hasBriefDocuemtnation = not isDocumentationEmpty(item.briefDescription)
	local hasDetailedDocuemtnation = not isDocumentationEmpty(item.detailedDescription)

	item.hasDocumentation = hasBriefDocuemtnation or hasDetailedDocuemtnation
	if not item.hasDocumentation then
		return false
	end

	if hasDetailedDocuemtnation then
		local text = getItemInternalDocumentation(item)

		item.isSubGroupHead = string.match(text, ":subgroup:") ~= nil
		if item.isSubGroupHead then
			item.subGroupSlaveArray = {}
		end
	end

	return not compound or not item.group or item.group == compound
end

function prepareItemArrayDocumentation(itemArray, compound)

	local hasDocumentation = false
	local subGroupHead = nil

	for i = 1, #itemArray do
		local item = itemArray[i]

		local result = prepareItemDocumentation(item, compound)
		if result then
			hasDocumentation = true
			if item.isSubGroupHead then
				subGroupHead = item
			else
				subGroupHead = nil
			end
		elseif subGroupHead then
			table.insert(subGroupHead.subGroupSlaveArray, item)
			item.subGroupHead = subGroupHead
		end
	end

	return hasDocumentation
end

function isItemInCompoundDetails(item, compound)
	if not item.hasDocumentation then
		return false
	end

	return not item.group or item.group == compound
end

function cmpIds(i1, i2)
	return i1.id < i2.id
end

function cmpNames(i1, i2)
	return i1.name < i2.name
end

function cmpTitles(i1, i2)
	return i1.title < i2.title
end

if not SORT_GROUPS_BY then
	SORT_GROUPS_BY = "title"
end

function cmpGroups(g1, g2)
	s1 = getItemInternalDocumentation(g1)
	s2 = getItemInternalDocumentation(g2)

	o1 = string.match(s1, ":grouporder%(([^)]*)%)")
	o2 = string.match(s2, ":grouporder%(([^)]*)%)")

	if o1 and o2 then
		return tonumber(o1) < tonumber(o2)
	else
		return g1[SORT_GROUPS_BY] < g2[SORT_GROUPS_BY]
	end
end

-------------------------------------------------------------------------------
