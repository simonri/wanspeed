import uuid

def convert_cond(cond):
  out = []
  for c in cond:
    temp = c[1].copy()
    model_conds = temp.get("model_conds", {})
    if c[0] is not None:
      temp["cross_attn"] = c[0]
    temp["model_conds"] = model_conds
    temp["uuid"] = uuid.uuid4()
    out.append(temp)
  return out
