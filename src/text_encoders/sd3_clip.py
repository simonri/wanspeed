def t5_xxl_detect(state_dict, prefix=""):
  out = {}
  t5_key = "{}encoder.final_layer_norm.weight".format(prefix)
  if t5_key in state_dict:
    out["dtype_t5"] = state_dict[t5_key].dtype

  scaled_fp8_key = "{}scaled_fp8".format(prefix)
  if scaled_fp8_key in state_dict:
    out["t5xxl_scaled_fp8"] = state_dict[scaled_fp8_key].dtype

  return out
