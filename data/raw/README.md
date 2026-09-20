# Raw data

The course-provided download list contained only XAI image datasets, so the
PR03-01 promoter arrays were obtained from the public repository accompanying
Lei et al., *Combining diffusion and transformer models for enhanced promoter
synthesis and strength prediction in deep learning* (mSystems, 2025):

https://github.com/LX2004/promoter/tree/main/data

Downloaded on 2026-09-16 without modification:

- `promoter.npy`: 11,884 E. coli promoter sequences, each 50 bp.
  SHA-256: `e71911452431acb620478f0de5ae896481be5618e7a7f9712125cd98cb8e37ab`
- `gene_expression.npy`: 11,884 positive expression-strength labels.
  SHA-256: `a1426fa5e5806fceff2c041218b781b6dc1f0cc84b39cf4ef626d02965fd9ce5`

Raw data files are ignored by Git. Keep these files unchanged so experiment
metadata and checksums remain reproducible.
