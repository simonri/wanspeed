class CLIPTextEncode:
  def encode(self, clip, text):
    if clip is None:
      raise RuntimeError(
        "ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model."
      )
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)