# Models

The weights are assets of the repository's releases, not files in it.

| Release | What it holds | Read by |
|---|---|---|
| [`d-lsa-weights-v1`](https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/d-lsa-weights-v1) | the distilled maps of D-LSA for Qwen3.8-27B and GLM-5.3-Flash | every `lsa_*` served config |
| [`n-lsa-weights-v1`](https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/n-lsa-weights-v1) | the checkpoints of the ten 128k decays of the native comparison, N-LSA and its baselines | `lsa native evaluate` |

```bash
gh release download d-lsa-weights-v1 -R Prefrontal-AI/latent-sparse-attention --pattern '*.pt' --dir "$LSA_ROOT/models/maps"
```

A native checkpoint is split in parts below the 2 GiB limit of a release asset.
[Weights](../README.md#weights) in the main README says how to join and unpack one, and which config evaluates each.
