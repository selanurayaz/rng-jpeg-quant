# RNG-Based JPEG Quantization Table Experiment

In this study, a simple pseudo-random number generator (XORSHIFT32) was implemented
to analyze the role of randomness in compression-related systems.

Rather than directly encrypting data, the RNG is used to generate an alternative
8x8 quantization table by applying controlled perturbations to the standard JPEG
luminance quantization table.

## Method
- Implemented a PRNG (XORSHIFT32)
- Generated a new quantization-like table using RNG
- Applied JPEG compression using OpenCV
- Since OpenCV does not allow custom quantization tables directly, the aggressiveness
  of the generated table was mapped to the JPEG quality parameter
- Compression performance was evaluated using file size and PSNR

## Results
The RNG-derived configuration achieved a smaller file size while maintaining
comparable PSNR values, indicating a favorable compression-quality trade-off.

## Outputs
- `results/summary.md`: quantitative comparison
- `results/*_std_*.jpg`: standard JPEG outputs
- `results/*_rng_*.jpg`: RNG-derived outputs
- `results/quant_tables_*.png`: visualization of quantization tables

## Technologies
Python, NumPy, OpenCV, scikit-image

## Security Perspective
Although the implemented RNG is not cryptographically secure,
the experiment demonstrates how randomness quality directly affects
systems that rely on probabilistic behavior.
Weak or predictable randomness can negatively impact both
security-related applications and multimedia processing pipelines.

## Simple Encryption Demonstration
A minimal XOR-based encryption example is included to demonstrate
how a pseudo-random number generator (PRNG) can be used to generate
a keystream for encryption.

This example is provided **for educational purposes only** and is
**not cryptographically secure**. It is intended to highlight the
importance of randomness quality in security-related systems.


