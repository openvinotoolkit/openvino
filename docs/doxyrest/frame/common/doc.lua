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

function concatDocBlockContents(s1, s2)
	local length1 = string.len(s1)
	local length2 = string.len(s2)

	if length1 == 0 then
		return s2
	elseif length2 == 0 then
		return s1
	end

	if string.match(s1, "[%s[{(<]$") or string.match(s2, "^[%s%]})>%.,;!?]") then
		return s1 .. s2
	else
		return s1 .. " " .. s2
	end
end

function getCodeDocBlockContents(block, context)
	context.codeBlockKind = block.blockKind
	local s = getDocBlockListContentsImpl(block.childBlockList, context)
	context.codeBlockKind = nil

	return s
end

function getDocBlockText(block, context)
	local text = block.text
	local childContents = getDocBlockListContentsImpl(block.childBlockList, context)

	if not context.codeBlockKind then
		if ESCAPE_ASTERISKS then
			text = string.gsub(text, "%*", "\\*")
			-- workaround for doxygen
			text = string.gsub(text, "%\\%\\%*", "\\*")
		end

		if ESCAPE_PIPES then
			text = string.gsub(text, "|", "\\|")
		end

		if ESCAPE_TRAILING_UNDERSCORES then
			text = string.gsub(text .. " ", "_([%s%p%c])", "\\_%1")
		end

		text = trimWhitespace(text)
		text = concatDocBlockContents(text, childContents)
	else
		text = text .. childContents
	end

	return text
end

-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

-- handling of different block kinds

function formatDocBlock_computeroutput(block, context)
	if context.codeBlockKind then
		return block.text .. getDocBlockListContentsImpl(block.childBlockList, context)
	end

	local code = block.text .. getCodeDocBlockContents(block, context)
	if not string.find(code, "\n") then
		return "``" .. trimWhitespace(code) .. "``"
	end

	code = replaceCommonSpacePrefix(code, "\t")
	code = trimTrailingWhitespace(code)
	return "\n\n.. code-block:: none\n\n" .. code
end

function formatDocBlock_programlisting(block, context)
	local code = getCodeDocBlockContents(block, context)
	code = replaceCommonSpacePrefix(code, "\t")
	code = trimTrailingWhitespace(code)

	return "\n\n.. ref-code-block:: " .. LANGUAGE .. "\n\n" .. code .. "\n\n"
end

function formatDocBlock_preformatted(block, context)
	local code = getCodeDocBlockContents(block, context)
	code = replaceCommonSpacePrefix(code, "\t")
	code = trimTrailingWhitespace(code)

	-- raw seems like a better approach, but need to figure something out with indents
	return "\n\n.. code-block:: none\n\n" .. code .. "\n\n"
end

function formatDocBlock_formula(block, context)
	local code = getCodeDocBlockContents(block, context)
	local isInline = string.sub(code, 1, 1) == "$"

	-- take away framing tokens
	code = string.gsub(code, "^\\?[$%[]", "")
	code = string.gsub(code, "\\?[$%]]$", "")

	if isInline then
		code = trimWhitespace(code)
		return ":math:`" .. code .. "`"
	end

	code = replaceCommonSpacePrefix(code, "\t")
	code = trimTrailingWhitespace(code)

	-- raw seems like a better approach, but need to figure something out with indents
	return "\n\n.. math::\n\n" .. code .. "\n\n"
end

function formatDocBlock_verbatim(block, context)
	if not VERBATIM_TO_CODE_BLOCK then
		return getDocBlockText(block, context)
	end

	local code = getCodeDocBlockContents(block, context)
	code = replaceCommonSpacePrefix(code, "\t")
	code = trimTrailingWhitespace(code)

	-- probably also need to assign some div class
	return "\n\n.. code-block:: " .. VERBATIM_TO_CODE_BLOCK .. "\n\n" .. code .. "\n\n"
end

function formatDocBlock_list(block, context, bullet)
	local prevBullet = context.listItemBullet
	context.listItemBullet = bullet

	local s = getDocBlockListContentsImpl(block.childBlockList, context)
	s = "\n\n" .. trimTrailingWhitespace(s) .. "\n\n"

	context.listItemBullet = prevBullet

	return s
end

function formatDocBlock_variablelist(block, context)
	local prevList = context.dlList
	context.dlList = {}

	getDocBlockListContentsImpl(block.childBlockList, context)

	local s =
		"\n\n.. list-table::\n" ..
		"\t:widths: 20 80\n\n"

	for i = 1, #context.dlList do
		s = s .. "\t*\n"

		local entry = context.dlList[i]
		local title = replaceCommonSpacePrefix(entry.title, "\t\t  ")
		local description = replaceCommonSpacePrefix(entry.description, "\t\t  ")

		s = s .. "\t\t- " .. trimLeadingWhitespace(title) .. "\n\n"
		s = s .. "\t\t- " .. trimLeadingWhitespace(description) .. "\n\n"
	end

	context.dlList = prevList

	return s
end

function formatDocBlock_ref(block, context)
	local text = getDocBlockText(block, context)
	text = string.gsub(text, "<", "\\<") -- escape left chevron
	return ":ref:`" .. text .. " <doxid-" .. block.id .. ">`"
end

function formatDocBlock_anchor(block, context)
	local text = getDocBlockText(block, context)
	return ":target:`doxid-" .. block.id .. "`" .. text
end

function formatDocBlock_image(block, context)
	if block.imageKind == 'html' then
		local s = "\n\n.. image:: " .. block.name .. "\n"
		if block.width ~= 0 then
			s = s .. "\t:width: " .. block.width .. "\n"
		end
		if block.height ~= 0 then
			s = s .. "\t:height: " .. block.height .. "\n"
		end
		if block.text ~= "" then
			s = s .. "\t:alt: " .. block.text .. "\n"
		end
		return s .. "\n"
	end
	return ''
end

function formatDocBlock_font(block, context, token)
	local text = getDocBlockText(block, context)

	if not string.match(text, "[\n`*]") then -- single line, no inline markup
		return token .. text .. token
	end

	return text
end

function formatDocBlock_heading(block, context)
	local text = getDocBlockText(block, context)

	if HEADING_TO_RUBRIC or not text or text == "" then
		return "\n\n.. rubric:: " .. text .. "\n\n"
	else
		return "\n\n" .. getTitle(text, block.level + 1) .. "\n\n"
	end
end

function formatDocBlock_varlistentry(block, context)
	if not context.dlList then
		error("unexpected <varlistentry>")
	end

	local text = getDocBlockText(block, context)
	local count = #context.dlList
	local entry = {}
	entry.title = trimWhitespace(text)
	entry.description = ""
	context.dlList[count + 1] = entry

	return ""
end

function formatDocBlock_listitem(block, context)
	local s = ""
	local text = getDocBlockText(block, context)

	if context.dlList then
		local count = #context.dlList
		local entry = context.dlList[count]
		if entry then
			entry.description = trimWhitespace(text)
		end
	else
		if not context.listItemBullet then
			error("unexpected <listitem>")
		end

		s = context.listItemBullet .. " "
		local indent = string.rep(' ', string.len(s))

		text = replaceCommonSpacePrefix(text, indent)
		text = trimWhitespace(text)

		s = s .. text .. "\n\n"
	end

	return s
end

function formatDocBlock_para(block, context)
	local s = getDocBlockText(block, context)
	s = trimWhitespace(s)

	if s ~= "" then
		s = s .. "\n\n"
	end

	return s
end

function formatDocBlock_parametername(block, context)
	local text = getDocBlockText(block, context)
	text = trimWhitespace(text)

	if not context.paramSection then
		context.paramSection = {}
	end

	local i = #context.paramSection + 1

	context.paramSection[i] = {}
	context.paramSection[i].name = text
	context.paramSection[i].description = ""

	return ""
end

function formatDocBlock_parameterdescription(block, context)
	local text = getDocBlockText(block, context)
	text = trimWhitespace(text)

	if string.find(text, "\n") then
		text = "\n" .. replaceCommonSpacePrefix(text, "\t\t  ") -- add paramter table offset "- "
	end

	if context.paramSection then
		local count = #context.paramSection
		context.paramSection[count].description = text
	end

	return ""
end

function formatDocBlock_sect(block, context, level)
	local s = "\n\n"
	local text = getDocBlockText(block, context)

	if block.id and block.id ~= "" then
		s = s .. ".. _doxid-" .. block.id .. ":\n\n"
	end

	if block.title and block.title ~= "" then
		if SECTION_TO_RUBRIC then
			s = s .. ".. rubric:: " .. block.title .. ":"
		else
			s = s .. getTitle(block.title, level + 1)
		end
	end

	return s .. "\n\n" .. text .. "\n\n"
end

function formatDocBlock_simplesect(block, context)
	local s = ""
	local text = getDocBlockText(block, context)

	if block.simpleSectionKind == "return" then
		if not context.returnSection then
			context.returnSection = {}
		end

		local count = #context.returnSection
		context.returnSection[count + 1] = text
	elseif block.simpleSectionKind == "see" then
		if not context.seeSection then
			context.seeSection = {}
		end

		local count = #context.seeSection
		context.seeSection[count + 1] = text
	else
		s = text
	end

	return s
end

function formatDocBlock_ulink(block, context)
	return "`" .. block.text .. " <" .. block.url .. ">`__"
end

function formatDocBlock_table(b, context)
	local blockList = b.childBlockList
	local tbl = {}
	local maxwidths = {}

	for i = 1, #blockList do
		local block = blockList [i]
		if block.blockKind == 'row' then
			local rowblock = block.childBlockList
			local row = {}
			for rownum = 1, #rowblock do
				if rowblock[rownum].blockKind == 'entry' then
					local entryblock = rowblock[rownum].childBlockList
					for entry = 1, #entryblock do
						if entryblock[entry].blockKind == 'para' then
							local t = ''
							local parablock = entryblock[entry].childBlockList
							for para = 1, #parablock do
								if parablock[para].blockKind == 'ref' then
									local blockText = getDocBlockText(parablock[para], context)
									blockText = string.gsub(blockText, "<", "\\<") -- escape left chevron
									t = t .. ":ref:`" .. blockText .. " <doxid-" .. parablock[para].id .. ">` "
								elseif parablock[para].blockKind == 'bold' then
									local blockText = getDocBlockText(parablock[para], context)
									t = t .. '**' .. blockText .. '** '
								elseif parablock[para].blockKind == 'emphasis' then
									local blockText = getDocBlockText(parablock[para], context)
									t = t .. '*' .. blockText .. '* '
								elseif parablock[para].blockKind == 'ulink' then
									t = t .. "`" .. parablock[para].text .. " <" .. parablock[para].url .. ">`__ "
								elseif parablock[para].blockKind == 'formula' then
										local code = getCodeDocBlockContents(parablock[para], context)
										-- take away framing tokens
										code = string.gsub(code, "^\\?[$%[]", "")
										code = string.gsub(code, "\\?[$%]]$", "")
										code = trimWhitespace(code)
										t = t .. ":math:`" .. code .. "` "
								elseif parablock[para].blockKind == 'computeroutput' then
									local code = getCodeDocBlockContents(parablock[para], context)
									t = t .. "``" .. code .. "`` "
								else
									moreT = trimWhitespace(parablock[para].text)
									if string.len(moreT) > 0 then
										moreT = moreT .. ' '
									end
									t = t .. moreT
								end
							end
							local tlen = string.len(t)
							table.insert(row, t)
							if maxwidths[#row] == nil or maxwidths[#row] < tlen then
								maxwidths[#row] = tlen
							end
						end
					end
				end
			end
			table.insert(tbl, row)
		end
	end

	if #tbl == 0 then
		return ''
	end

	local s = '.. list-table::\n'
	local s = s .. '    :header-rows: 1\n\n'
	for r = 1, #tbl do
		for c = 1, #tbl[r] do
			if c == 1 then
				s = s .. '    * - '
			else
				s = s .. '      - '
			end
			tblrc = trimWhitespace(tbl[r][c])
			s = s .. tblrc .. '\n'
			-- s = s .. tblrc .. string.rep(' ', 2 + maxwidths[c] - string.len(tblrc))
		end
	end
	return '\n\n' .. s .. '\n'
end


function formatDocBlock_graph(block, context, graphtype)
	local code = getCodeDocBlockContents(block, context)
	code = replaceCommonSpacePrefix(code, "\t")
	code = trimTrailingWhitespace(code)

	return "\n\n.. " .. graphtype .. "::\n\n" .. code .. "\n\n"
end

function formatDocBlock_blockquote(block, context)
	local text = getDocBlockText(block, context)
	if text.find(text,'**NOTE**') then
		text = string.gsub(text, '^%s*%*%*NOTE%*%*%s*:%s*', '')
		return "\n\n.. note:: " .. text .. "\n\n"
	end
	if text.find(text,'**IMPORTANT**') then
		text = string.gsub(text, '^%s*%*%*IMPORTANT%*%*%s*:%s*', '')
		return "\n\n.. warning:: " .. text .. "\n\n"
	end
	if text.find(text,'**TIP**') then
		text = string.gsub(text, '^%s*%*%*TIP%*%*%s*:%s*', '')
		return "\n\n.. tip:: " .. text .. "\n\n"
	end
	return "\t" .. text
end

function formatDocBlock_sphinxdirective(block, context)
	local code = getCodeDocBlockContents(block, context)
	return "\n\n" .. code .. "\n\n"
end

function formatDocBlock_sphinxtabset(block, context)
	return "\n\n" .. ".. raw:: html\n\n   <div class='sphinxtabset'>" .. "\n\n"
end

function formatDocBlock_endsphinxtabset(block, context)
	return "\n\n" .. ".. raw:: html\n\n   </div>" .. "\n\n"
end

function formatDocBlock_sphinxtab(block, context)
	local code = getCodeDocBlockContents(block, context)
	return "\n\n" .. '.. raw:: html\n\n   <div class="sphinxtab" data-sphinxtab-value="' .. code .. '">' .. "\n\n"
end

function formatDocBlock_endsphinxtab(block, context)
	return "\n\n" .. ".. raw:: html\n\n   </div>" .. "\n\n"
end


-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

g_blockKindFormatMap =
{
	["computeroutput"]       = formatDocBlock_computeroutput,
	["programlisting"]       = formatDocBlock_programlisting,
	["preformatted"]         = formatDocBlock_preformatted,
	["formula"]              = formatDocBlock_formula,
	["verbatim"]             = formatDocBlock_verbatim,
	["itemizedlist"]         = function(b, c) return formatDocBlock_list(b, c, "*") end,
	["orderedlist"]          = function(b, c) return formatDocBlock_list(b, c, "#.") end,
	["variablelist"]         = formatDocBlock_variablelist,
	["linebreak"]            = function(b, c) return "\n\n" end,
	["ref"]                  = formatDocBlock_ref,
	["anchor"]               = formatDocBlock_anchor,
	["image"]                = formatDocBlock_image,
	["bold"]                 = function(b, c) return formatDocBlock_font(b, c, "**") end,
	["emphasis"]             = function(b, c) return formatDocBlock_font(b, c, "*") end,
	["sp"]                   = function(b, c) return " " end,
	["varlistentry"]         = formatDocBlock_varlistentry,
	["listitem"]             = formatDocBlock_listitem,
	["heading"]              = formatDocBlock_heading,
	["para"]                 = formatDocBlock_para,
	["parametername"]        = formatDocBlock_parametername,
	["parameterdescription"] = formatDocBlock_parameterdescription,
	["sect1"]                = function(b, c) return formatDocBlock_sect(b, c, 1) end,
	["sect2"]                = function(b, c) return formatDocBlock_sect(b, c, 2) end,
	["sect3"]                = function(b, c) return formatDocBlock_sect(b, c, 3) end,
	["sect4"]                = function(b, c) return formatDocBlock_sect(b, c, 4) end,
	["simplesect"]           = formatDocBlock_simplesect,
	["ulink"]                = formatDocBlock_ulink,
	["table"]                = formatDocBlock_table,
	["dot"]                  = function(b, c) return formatDocBlock_graph(b, c, "graphviz") end,
	["plantuml"]             = function(b, c) return formatDocBlock_graph(b, c, "uml") end,
	["msc"]                  = function(b, c) return formatDocBlock_graph(b, c, "msc") end,
	["blockquote"]           = formatDocBlock_blockquote,
	["sphinxdirective"]      = formatDocBlock_sphinxdirective,
	["sphinxtabset"]         = formatDocBlock_sphinxtabset,
	["endsphinxtabset"]      = formatDocBlock_endsphinxtabset,
	["sphinxtab"]            = formatDocBlock_sphinxtab,
	["endsphinxtab"]         = formatDocBlock_endsphinxtab
}

function getDocBlockContents(block, context)
	format = g_blockKindFormatMap[block.blockKind]
	if format then
		return format(block, context)
	else
		return getDocBlockText(block, context)
	end
end

-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

function getDocBlockListContentsImpl(blockList, context)
	local s = ""

	for i = 1, #blockList do
		local block = blockList[i]
		if block.blockKind ~= "internal" then
			local blockContents = getDocBlockContents(block, context)
			if not context.codeBlockKind then
				s = concatDocBlockContents(s, blockContents)
			else
				s = s .. blockContents
			end
		end
	end

	return s
end

function getDocBlockListContents(blockList)
	local context = {}
	local s = getDocBlockListContentsImpl(blockList, context)

	if context.paramSection then
		s = s .. "\n\n.. rubric:: Parameters:\n\n"
		s = s .. ".. list-table::\n"
		s = s .. "\t:widths: 20 80\n\n"

		for i = 1, #context.paramSection do
			s = s .. "\t*\n"
			s = s .. "\t\t- " .. context.paramSection[i].name .. "\n\n"
			s = s .. "\t\t- " .. context.paramSection[i].description .. "\n\n"
		end
	end

	if context.returnSection then
		s = s .. "\n\n.. rubric:: Returns:\n\n"

		for i = 1, #context.returnSection do
			s = s .. context.returnSection[i]
		end
	end

	if context.seeSection then
		s = s .. "\n\n.. rubric:: See also:\n\n"

		for i = 1, #context.seeSection do
			s = s .. context.seeSection[i]
		end
	end

	s = trimTrailingWhitespace(s)

	return replaceCommonSpacePrefix(s, "")
end

function getSimpleDocBlockListContents(blockList)
	local s = ""

	for i = 1, #blockList do
		local block = blockList[i]
		s = s .. block.text .. getSimpleDocBlockListContents(block.childBlockList)
	end

	return s
end

function isDocumentationEmpty(description)
	if description.isEmpty then
		return true
	end

	local text = getDocBlockListContents(description.docBlockList)
	return string.len(text) == 0
end

-------------------------------------------------------------------------------
