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

function printTable(t)
	for key, value in pairs(t) do
		print(key, " -> ", value)
	end
end

function concatenateTables(t1, t2)
	local j = #t1 + 1
	for i = 1, #t2 do
		t1 [j] = t2 [i]
		j = j + 1
	end
end

function filterArray(a, filter)
	if next(a) == nil then
		return
	end

	for i = #a, 1, -1 do
		if not filter(a[i]) then
			table.remove(a, i)
		end
	end
end

--------------------------------------------------------------------------------
