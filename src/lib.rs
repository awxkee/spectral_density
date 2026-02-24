/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#![allow(clippy::too_many_arguments)]
mod err;
mod mla;
mod welch;

pub use detrend::DetrendingMethod;
pub use err::SpectralError;
use num_traits::{AsPrimitive, Float, MulAdd, NumCast};
use std::fmt::Debug;
use std::ops::{AddAssign, MulAssign};
use std::sync::Arc;
use zaft::{FftExecutor, Zaft};

pub struct Welch<'a, T> {
    pub input: &'a [T],
    pub fs: f64,
    pub window: WelchWindow,
    pub nperseg: usize,
    pub noverlap: usize,
    pub fft_size: Option<usize>,
    pub detrend: DetrendingMethod,
    pub scaling: ScalingMethod,
}

impl<'a> Welch<'a, f32> {
    pub fn new(input: &'a [f32]) -> Self {
        let nperseg = 256.min(input.len());

        Self {
            input,
            fs: 1.0,
            window: WelchWindow::Hann,
            nperseg,
            noverlap: nperseg / 2,
            fft_size: None,
            detrend: DetrendingMethod::Constant,
            scaling: ScalingMethod::Density,
        }
    }

    pub fn fs(mut self, fs: f64) -> Self {
        self.fs = fs;
        self
    }

    pub fn window(mut self, window: WelchWindow) -> Self {
        self.window = window;
        self
    }

    pub fn nperseg(mut self, n: usize) -> Self {
        self.nperseg = n;
        self
    }

    pub fn noverlap(mut self, n: usize) -> Self {
        self.noverlap = n;
        self
    }

    pub fn nfft(mut self, n: usize) -> Self {
        self.fft_size = Some(n);
        self
    }

    pub fn detrend(mut self, method: DetrendingMethod) -> Self {
        self.detrend = method;
        self
    }

    pub fn scaling(mut self, scaling: ScalingMethod) -> Self {
        self.scaling = scaling;
        self
    }
}

/// Represents the available window functions for use in Welch's method.
#[derive(Debug, Copy, Clone, PartialEq, Hash, Ord, PartialOrd, Eq)]
pub enum WelchWindow {
    /// The Hann window function.
    Hann,
    /// The Hamming window function.
    Hamming,
    /// The Blackman window function.
    Blackman,
}

#[derive(Debug, Copy, Clone, PartialEq, Hash, Ord, PartialOrd, Eq)]
pub enum ScalingMethod {
    /// Returns the Power Spectral Density (PSD). Units are Power / Hz.
    Density,
    /// Returns the Power Spectrum. Units are Power (PSD * fs).
    Spectrum,
}

use crate::welch::welch_impl;
pub use welch::WelchResult;

/// Computes the Power Spectral Density (PSD) or Power Spectrum of a signal
/// using Welch's method for single-precision floating-point (`f32`) data.
///
/// This method averages periodograms of overlapping segments to reduce variance.
///
/// # Parameters
///
/// * `input`: The time-series data array.
/// * `fs`: Optional sampling frequency (default: 1.0).
/// * `window`: Optional window function type (default: `WelchWindow::Hann`).
/// * `nperseg`: Optional length of each segment (default: `min(256, input.len())`).
/// * `noverlap`: Optional number of points to overlap between segments (default: `nperseg / 2`).
/// * `nfft`: Optional length of the FFT (default: `nperseg`).
/// * `detrend`: Optional detrending method for each segment (default: `DetrendingMethod::Constant`).
/// * `scaling`: Optional scaling method for the output (default: `ScalingMethod::Density`).
///
/// # Returns
///
/// A `Result` containing the frequencies and the computed power spectral density/spectrum,
/// or a `SpectralError` if validation fails.
pub fn welch_f32(welch: &Welch<f32>) -> Result<WelchResult<f32>, SpectralError> {
    welch_impl(welch)
}

/// Computes the Power Spectral Density (PSD) or Power Spectrum of a signal
/// using Welch's method for double-precision floating-point (`f64`) data.
///
/// This method averages periodograms of overlapping segments to reduce variance.
///
/// # Parameters
///
/// (Same as `welch_f32`)
///
/// # Returns
///
/// A `Result` containing the frequencies and the computed power spectral density/spectrum,
/// or a `SpectralError` if validation fails.
pub fn welch_f64(welch: &Welch<f64>) -> Result<WelchResult<f64>, SpectralError> {
    welch_impl(welch)
}

pub(crate) trait WelchSample:
    Float
    + NumCast
    + Debug
    + WindowGenerator
    + MulAdd<Self, Output = Self>
    + 'static
    + AsPrimitive<f64>
    + MulAssign
    + AddAssign
{
    fn detrend(
        samples: &[Self],
        detrending_method: DetrendingMethod,
    ) -> Result<Vec<Self>, SpectralError>;
    fn make_fft(size: usize) -> Result<Arc<dyn FftExecutor<Self> + Send + Sync>, SpectralError>;
}

pub(crate) trait WindowGenerator: Sized {
    fn get_window(window: WelchWindow, n: usize) -> Vec<Self>;
}

impl WindowGenerator for f32 {
    fn get_window(window: WelchWindow, n: usize) -> Vec<Self> {
        match window {
            WelchWindow::Hann => pxwindow::Pxwindow::hann_f32(n),
            WelchWindow::Hamming => pxwindow::Pxwindow::hamming_f32(n),
            WelchWindow::Blackman => pxwindow::Pxwindow::blackman_f32(n),
        }
    }
}

impl WindowGenerator for f64 {
    fn get_window(window: WelchWindow, n: usize) -> Vec<Self> {
        match window {
            WelchWindow::Hann => pxwindow::Pxwindow::hann_f64(n),
            WelchWindow::Hamming => pxwindow::Pxwindow::hamming_f64(n),
            WelchWindow::Blackman => pxwindow::Pxwindow::blackman_f64(n),
        }
    }
}

impl WelchSample for f64 {
    fn detrend(
        samples: &[Self],
        detrending_method: DetrendingMethod,
    ) -> Result<Vec<Self>, SpectralError> {
        detrend::detrend_f64(samples, detrending_method)
            .map_err(|x| SpectralError::DetrendingError(x.to_string()))
    }

    fn make_fft(size: usize) -> Result<Arc<dyn FftExecutor<Self> + Send + Sync>, SpectralError> {
        Zaft::make_forward_fft_f64(size).map_err(|x| SpectralError::FftError(x.to_string()))
    }
}

impl WelchSample for f32 {
    fn detrend(
        samples: &[Self],
        detrending_method: DetrendingMethod,
    ) -> Result<Vec<Self>, SpectralError> {
        detrend::detrend_f32(samples, detrending_method)
            .map_err(|x| SpectralError::DetrendingError(x.to_string()))
    }

    fn make_fft(size: usize) -> Result<Arc<dyn FftExecutor<Self> + Send + Sync>, SpectralError> {
        Zaft::make_forward_fft_f32(size).map_err(|x| SpectralError::FftError(x.to_string()))
    }
}
