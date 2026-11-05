wrk.method = "POST"
wrk.path   = "/query"
wrk.headers["Content-Type"] = "application/json"

local questions = {
  "Luật Giao thông đường bộ yêu cầu người lái xe làm gì trước khi chuyển làn?",
  "Các bước xử lý khi gặp chướng ngại vật bất ngờ trên đường là gì?",
  "Quy định về tốc độ tối đa trong khu dân cư ban ngày là bao nhiêu?",
  "Điều kiện kỹ thuật của hệ thống phanh an toàn là gì?",
  "Những trường hợp nào phải nhường quyền ưu tiên cho xe buýt?"
}

local session_id = "wrk-load"

local function escape(str)
  return str:gsub('"', '\\"')
end

math.randomseed(os.time())

function request()
  local idx = math.random(#questions)
  local body = string.format('{"question":"%s","session_id":"%s"}', escape(questions[idx]), session_id)
  return wrk.format(nil, nil, body)
end
