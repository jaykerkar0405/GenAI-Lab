-- TeXperiment: inserts an optional "Code" section between Steps and Output.
--
-- In the Markdown front matter set:
--     code: code.py                (one file)
--     code: [main.c, utils.c]      (several files)
--     language: Python             (optional; guessed from the extension)
-- If `code` is missing, no Code section is created.
-- The code file is read by LaTeX directly, so it is never rewritten.

local ext_to_lang = {
  py = "Python", c = "C", h = "C", cpp = "C++", cc = "C++", hpp = "C++",
  java = "Java", sql = "SQL", sh = "bash", js = "JavaScript", html = "HTML",
  xml = "XML", m = "Matlab", r = "R", v = "Verilog", vhd = "VHDL",
  php = "PHP", pl = "Perl", rb = "Ruby", go = "Go", cs = "[Sharp]C",
  tex = "TeX", css = "CSS", txt = nil,
}

local function esc(s)
  s = s:gsub("\\", "\\textbackslash{}")
  s = s:gsub("([_%%&#%$%^~{}])", function(c)
    if c == "^" then return "\\^{}" end
    if c == "~" then return "\\~{}" end
    return "\\" .. c
  end)
  return s
end

local function guess_lang(file, override)
  if override and override ~= "" then return override end
  local ext = file:match("%.([%w]+)$")
  if ext then return ext_to_lang[ext:lower()] end
  return nil
end

local function is_output_heading(block)
  if block.t ~= "Header" or block.level ~= 1 then return false end
  local t = pandoc.utils.stringify(block.content):lower()
  return t:find("output") ~= nil or t:find("screenshot") ~= nil or t:find("result") ~= nil
end

local function is_conclusion_heading(block)
  if block.t ~= "Header" or block.level ~= 1 then return false end
  return pandoc.utils.stringify(block.content):lower():find("conclusion") ~= nil
end

function Pandoc(doc)
  local meta = doc.meta
  if not meta.code then return nil end

  local files = {}
  if meta.code.t == "MetaList" then
    for _, item in ipairs(meta.code) do
      table.insert(files, pandoc.utils.stringify(item))
    end
  else
    table.insert(files, pandoc.utils.stringify(meta.code))
  end
  if #files == 0 then return nil end

  local override = meta.language and pandoc.utils.stringify(meta.language) or nil

  local new_blocks = { pandoc.Header(1, pandoc.Str("Code")) }
  for _, f in ipairs(files) do
    local lang = guess_lang(f, override)
    local opts = "caption={\\texttt{" .. esc(f) .. "}}"
    if lang then opts = "language=" .. lang .. "," .. opts end
    table.insert(new_blocks,
      pandoc.RawBlock("latex", "\\lstinputlisting[" .. opts .. "]{" .. f .. "}"))
  end

  -- find where to insert: before Output, else before Conclusion, else at the end
  local pos = nil
  for i, b in ipairs(doc.blocks) do
    if is_output_heading(b) then pos = i; break end
  end
  if not pos then
    for i, b in ipairs(doc.blocks) do
      if is_conclusion_heading(b) then pos = i; break end
    end
  end
  if not pos then pos = #doc.blocks + 1 end

  for k, b in ipairs(new_blocks) do
    table.insert(doc.blocks, pos + k - 1, b)
  end
  return doc
end
