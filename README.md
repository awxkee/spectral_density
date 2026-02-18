# Spectral Analysis Library (Welch's Method)

## 📖 Overview

This library provides an implementation of **Welch's method** for estimating the Power Spectral Density (PSD) and Power Spectrum of a time-series signal. Welch's method is a non-parametric technique that reduces the variance of the standard periodogram by averaging the modified periodograms of overlapping signal segments.

The core functionality is exposed through public wrappers for `f32` and `f64` data types.

## ✨ Key Features

* **Data Types:** Supports both single-precision (`f32`) and double-precision (`f64`) input.
* **Windowing:** Multiple window functions are available to minimize spectral leakage:
    * `WelchWindow::Hann`
    * `WelchWindow::Hamming`
    * `WelchWindow::Blackman`
* **Detrending:** Supports removing DC offsets (`DetrendingMethod::Constant`) or linear trends (`DetrendingMethod::Linear`) from each segment.
* **Scaling Control:** Configure the output as either:
    * **Power Spectral Density (`Density`):** Units of Power/Hz.
    * **Power Spectrum (`Spectrum`):** Units of Power.

## ⚙️ Public Functions and Parameters

The primary entry points are the type-specific functions: `welch_f32` and `welch_f64`.

-----

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.