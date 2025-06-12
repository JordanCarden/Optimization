# Composite Noise Model Report for Synthetic GFP Data

## 1. Noise Generation Method

Each synthetic measurement `y_t` is generated from the clean ODE output `x_t` by applying four noise components:

```text
δ_t ~ Normal(0, σ_δ²)
η_t ~ Normal(0, a + b·x_t)
ε_t = ρ·ε_{t-1} + η_t    (ε₀ = 0)
drift_t = drift_slope × t
y_t = (1 + δ_t)·x_t + ε_t + drift_t
```

- `a`: baseline noise variance
- `b`: signal-dependent noise coefficient
- `ρ`: autocorrelation coefficient for additive noise
- `σ_δ`: multiplicative noise scale (relative fluctuations)
- `drift_slope`: linear drift per time unit

## 2. Rationale for Model Components

- **Baseline & signal-dependent noise** (`a`, `b`)
  - Captures constant read/dark noise plus shot noise that increases with signal
- **Autocorrelated noise** (`ρ`)
  - Models temporal “stickiness” in measurement errors (e.g. instrument filter effects)
- **Multiplicative fluctuations** (`σ_δ`)
  - Accounts for percent-level errors (pipetting, light source stability)
- **Baseline drift** (`drift_slope`)
  - Simulates slow changes in baseline over long runs (temperature shifts, lamp warm‑up)

## 3. Advantages over Simple Gaussian Noise

A simple model uses

```text
y_t = x_t + ε_t   where ε_t ~ Normal(0, σ²)
```

which assumes independent, constant-variance errors. The composite model instead:

- Varies noise with signal intensity
- Introduces temporal correlation
- Includes percent-scale fluctuations
- Adds a slow baseline trend

These features align with real fluorescence measurements.

## 4. Parameter Selection and Justification

| Parameter               | Value                    | Justification                                                                                                                                                                                                  |
| ----------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **a**                   | 25                       | Baseline variance: SD≈5 AU; matches blank-well noise in plate readers ([cloudynights.com](https://www.cloudynights.com), [teledynevisionsolutions.com](https://www.teledynevisionsolutions.com)).              |
| **b**                   | 1.0                      | Shot noise: Poisson variance ≈ mean photon count; Fano factor≈1 ([Wikipedia](https://en.wikipedia.org/wiki/Shot_noise)).                                                                                       |
| **σ_δ** (`sigma-delta`) | 0.05 (5 %)               | Percent errors: pipetting and lamp fluctuations ~1–5 % CV ([capp.dk](https://capp.dk), [integra-biosciences.com](https://www.integra-biosciences.com)).                                                        |
| **ρ** (`rho`)           | 0.007                    | AR(1) correlation from low-pass filtering (τ≈2 min, Δt=10 min → ρ≈e^(–Δt/τ)≈0.007) ([Control Systems Academy](https://controlsystemsacademy.com), [Wikipedia](https://en.wikipedia.org/wiki/Low-pass_filter)). |
| **drift_slope**         | 0.01 AU/min (~0.6 AU/hr) | Baseline drift: ~0.6 AU per hour; consistent with long-term drift specs in fluorescence detectors ([support.waters.com](https://support.waters.com)).                                                          |

## References

1. CloudyNights discussion of read noise: https://www.cloudynights.com
2. Teledyne Vision Solutions camera noise guide: https://www.teledynevisionsolutions.com
3. Wikipedia on shot noise: https://en.wikipedia.org/wiki/Shot_noise
4. Common pipetting errors blog: https://capp.dk/blog/common-pipetting-errors
5. Integra Biosciences pipette accuracy: https://www.integra-biosciences.com
6. Control Systems Academy on low-pass filters: https://controlsystemsacademy.com
7. Wikipedia on low-pass filters: https://en.wikipedia.org/wiki/Low-pass_filter
8. Waters support note on detector drift: https://support.waters.com

---

_This report presents a noise model suitable for publication. Adjust any URLs or citations as needed._
