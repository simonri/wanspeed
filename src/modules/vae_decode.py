class VAEDecode:
  def decode(self, vae, samples):
    images = vae.decode(samples["samples"])
    if len(images.shape) == 5:  # Combine batches
      images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return (images,)
