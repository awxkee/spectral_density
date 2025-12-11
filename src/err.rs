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
use std::error::Error;
use std::fmt::Display;

#[derive(Debug)]
pub enum SpectralError {
    ZeroSizedInput,
    NegativeSamplingFrequency(f64),
    FftSizeIsTooSmall(usize, usize),
    NoOverlappingWindowLength(usize, usize),
    AmountOfSegmentsIsTooSmall,
    FrequenciesZeroBaseError,
    DetrendingError(String),
    FftError(String),
    /// Indicates a failure to allocate the memory required for the resulting vector.
    /// The associated value is the requested size (`usize`) of the allocation.
    Allocation(usize),
}

impl Display for SpectralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpectralError::ZeroSizedInput => f.write_str("Input size is zero"),
            SpectralError::NegativeSamplingFrequency(s) => f.write_fmt(format_args!(
                "Sampling frequency is negative: {s}, must be > 0"
            )),
            SpectralError::FftSizeIsTooSmall(s0, s1) => f.write_fmt(format_args!(
                "nfft must be at least as large as nperseg, got {s0} < {s1}"
            )),
            SpectralError::NoOverlappingWindowLength(s0, s1) => f.write_fmt(format_args!(
                "noverlap must be less than nperseg, got {s0} >= {s1}"
            )),
            SpectralError::AmountOfSegmentsIsTooSmall => {
                f.write_str("Not enough data points for given nperseg and noverlap")
            }
            SpectralError::FrequenciesZeroBaseError => f.write_str("n must be positive"),
            SpectralError::DetrendingError(s) => {
                f.write_fmt(format_args!("An error has occur while detrending: {s}"))
            }
            SpectralError::FftError(s) => {
                f.write_fmt(format_args!("An error has occur while FFT: {s}"))
            }
            SpectralError::Allocation(size) => {
                f.write_fmt(format_args!("Failed to allocate buffer with size {size}"))
            }
        }
    }
}

impl Error for SpectralError {}

macro_rules! try_vec {
    () => {
        Vec::new()
    };
    ($elem:expr; $n:expr) => {{
        let mut v = Vec::new();
        v.try_reserve_exact($n)
            .map_err(|_| crate::err::SpectralError::Allocation($n))?;
        v.resize($n, $elem);
        v
    }};
}

pub(crate) use try_vec;
