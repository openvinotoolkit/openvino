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

function getCompoundTocTree(compound)
	local s = ".. toctree::\n\t:hidden:\n\n"

	for i = 1, #compound.groupArray do
		local item = compound.groupArray[i]
		local fileName = getItemFileName(item)
		s = s .. "\t" .. fileName .. "\n"
	end

	for i = 1, #compound.namespaceArray do
		local item = compound.namespaceArray[i]
		if isTocTreeItem(compound, item) then
			local fileName = getItemFileName(item)
			s = s .. "\t" .. fileName .. "\n"
		end
	end

	for i = 1, #compound.enumArray do
		local item = compound.enumArray[i]
		if isTocTreeItem(compound, item) and not isUnnamedItem(item) then
			local fileName = getItemFileName(item)
			s = s .. "\t" .. fileName .. "\n"
		end
	end

	for i = 1, #compound.structArray do
		local item = compound.structArray[i]
		if isTocTreeItem(compound, item) then
			local fileName = getItemFileName(item)
			s = s .. "\t" .. fileName .. "\n"
		end
	end

	for i = 1, #compound.unionArray do
		local item = compound.unionArray[i]
		if isTocTreeItem(compound, item) then
			local fileName = getItemFileName(item)
			s = s .. "\t" .. fileName .. "\n"
		end
	end

	for i = 1, #compound.interfaceArray do
		local item = compound.interfaceArray[i]
		if isTocTreeItem(compound, item) then
			local fileName = getItemFileName(item)
			s = s .. "\t" .. fileName .. "\n"
		end
	end

	for i = 1, #compound.protocolArray do
		local item = compound.protocolArray[i]
		if isTocTreeItem(compound, item) then
			local fileName = getItemFileName(item)
			s = s .. "\t" .. fileName .. "\n"
		end
	end

	for i = 1, #compound.exceptionArray do
		local item = compound.exceptionArray[i]
		if isTocTreeItem(compound, item) then
			local fileName = getItemFileName(item)
			s = s .. "\t" .. fileName .. "\n"
		end
	end

	for i = 1, #compound.classArray do
		local item = compound.classArray[i]
		if isTocTreeItem(compound, item) then
			local fileName = getItemFileName(item)
			s = s .. "\t" .. fileName .. "\n"
		end
	end

	for i = 1, #compound.singletonArray do
		local item = compound.singletonArray[i]
		if isTocTreeItem(compound, item) then
			local fileName = getItemFileName(item)
			s = s .. "\t" .. fileName .. "\n"
		end
	end

	for i = 1, #compound.serviceArray do
		local item = compound.serviceArray[i]
		if isTocTreeItem(compound, item) then
			local fileName = getItemFileName(item)
			s = s .. "\t" .. fileName .. "\n"
		end
	end

	if compound.hasGlobalNamespace then
		s = s .. "\t" .. "global.rst" .. "\n"
	end

	return trimTrailingWhitespace(s)
end

function getGroupName(group)
	local s
	if string.len(group.title) ~= 0 then
		s = group.title
	else
		s = group.name
	end

	return s
end

function getGroupTree(group, indent)
	local s = ""

	if not indent then
		indent = ""
	end

	local name = getGroupName(group)
	name = string.gsub(name, "<", "\\<") -- escape left chevron

	s = "|\t" .. indent .. ":ref:`" .. name .. "<doxid-" ..group.id .. ">`\n"

	for i = 1, #group.groupArray do
		s = s .. getGroupTree(group.groupArray[i], indent .. "\t")
	end

	return s
end

function getPageTree(page, fileName, indent)
	local s = ""

	if not indent then
		indent = ""
	end

	local docName = string.gsub(fileName, "%.rst$", "")

	s = "|\t" .. indent .. ":doc:`" .. docName .. "`\n"

	for i = 1, #page.subPageArray do
		local subPage = page.subPageArray[i]
		s = s .. getPageTree(subPage, getItemFileName(subPage), indent .. "\t")
	end

	return s
end

--------------------------------------------------------------------------------
