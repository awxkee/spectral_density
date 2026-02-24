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
use crate::err::{SpectralError, try_vec};
use crate::mla::fmla;
use crate::{ScalingMethod, Welch, WelchSample, WelchWindow};
use detrend::DetrendingMethod;
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};

fn fftfreq<T: WelchSample>(n: usize, d: T) -> Result<Vec<T>, SpectralError>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
    i64: AsPrimitive<T>,
{
    if n == 0 {
        return Err(SpectralError::FrequenciesZeroBaseError);
    }

    let val = 1.0f64.as_() / (n.as_() * d);
    // Even case
    let arr_length = n / 2 + 1;
    let mut freq = try_vec![0f64.as_(); arr_length];
    for (i, dst) in freq.iter_mut().enumerate() {
        *dst = i.as_() * val;
    }
    Ok(freq)
}

pub struct WelchResult<T> {
    pub frequencies: Vec<T>,
    pub psd: Vec<T>,
}

pub(crate) fn welch_impl<T: WelchSample>(
    data: &Welch<'_, T>,
) -> Result<WelchResult<T>, SpectralError>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
    i64: AsPrimitive<T>,
{
    // Validate input
    if data.input.is_empty() {
        return Err(SpectralError::ZeroSizedInput);
    }

    // Default parameters
    let window_length = data.nperseg;
    let window_no_overlapping_length = data.noverlap;
    let fft_size = data.fft_size;
    let detrend_val = data.detrend;
    let scaling_val = data.scaling;

    // Validate parameters
    if data.fs <= 0.0 {
        return Err(SpectralError::NegativeSamplingFrequency(data.fs));
    }

    if fft_size < window_length {
        return Err(SpectralError::FftSizeIsTooSmall(fft_size, window_length));
    }

    if window_no_overlapping_length >= window_length {
        return Err(SpectralError::NoOverlappingWindowLength(
            window_no_overlapping_length,
            window_length,
        ));
    }

    // Create window function
    let win = T::get_window(data.window, window_length);
    let win_scale: T = {
        let mut x = T::zero();
        for &v in win.iter() {
            x = fmla(v, v, x);
        }
        x
    };
    let scale = 1.0f64.as_() / win_scale;

    // Determine number of segments
    let step = window_length - window_no_overlapping_length;
    #[allow(unknown_lints)]
    #[allow(clippy::manual_checked_ops)]
    let num_segments = if step > 0 {
        (data.input.len() - window_no_overlapping_length) / step
    } else {
        0
    };

    if num_segments < 1 {
        return Err(SpectralError::AmountOfSegmentsIsTooSmall);
    }

    // Calculate frequency bins
    let freqs = fftfreq(fft_size, 1.0f64.as_() / data.fs.as_())?;

    // Keep only positive frequencies
    let n_half = (fft_size / 2) + 1; // Handle both even and odd nfft
    let result_freqs: Vec<T> = freqs.into_iter().take(n_half).collect();

    // Initialize averaged periodogram
    let mut psd_avg = try_vec![0.0f64.as_(); n_half];

    let fft_executor = T::make_fft(fft_size)?;

    // Process each segment
    for i in 0..num_segments {
        let start = i * step;
        let end = start + window_length;

        if end > data.input.len() {
            break;
        }

        // Extract segment
        let segment = data.input[start..end].to_vec();

        // Detrend the segment
        let detrended = T::detrend(&segment, detrend_val)?;

        // Apply window
        let windowed: Vec<T> = detrended
            .iter()
            .zip(win.iter())
            .map(|(&x, &w)| x * w)
            .collect();

        // Zero-pad if needed
        let mut spectrum = windowed
            .iter()
            .map(|x| Complex::new(*x, 0.0f64.as_()))
            .collect::<Vec<_>>();
        if fft_size != spectrum.len() {
            spectrum.resize(fft_size, Complex::zero());
        }

        fft_executor
            .execute(&mut spectrum)
            .map_err(|x| SpectralError::FftError(x.to_string()))?;

        let freq_scaling: T = scale / (data.fs.as_());

        // Compute periodogram for this segment
        let mut segment_psd: Vec<T> = spectrum
            .iter()
            .take(n_half)
            .map(|&c| fmla(c.re, c.re, c.im * c.im) * freq_scaling)
            .collect();

        let two = 2.0f64.as_();

        // Loop from the 1st element (index 1) up to the second-to-last element (Nyquist or one before)
        // Note: For N=64, n_half=33. Indices 1..32 are non-DC, non-Nyquist.
        let end_index = if fft_size.is_multiple_of(2) {
            // Even FFT size: Nyquist is at n_half - 1. Loop up to but NOT including n_half - 1
            n_half - 1
        } else {
            // Odd FFT size: Nyquist does not exist. Loop up to n_half
            n_half
        };

        for j in 1..end_index {
            if j < segment_psd.len() {
                unsafe {
                    *segment_psd.get_unchecked_mut(j) *= two;
                }
            }
        }

        // Accumulate into average
        for (j, &psd) in segment_psd.iter().enumerate() {
            if j < psd_avg.len() {
                unsafe {
                    *psd_avg.get_unchecked_mut(j) += psd;
                }
            }
        }
    }

    // Normalize by number of segments
    let segment_scaling = 1f64.as_() / num_segments.as_();
    for psd in &mut psd_avg {
        *psd *= segment_scaling;
    }

    // Apply scaling
    let result_psd = match scaling_val {
        ScalingMethod::Density => psd_avg,
        ScalingMethod::Spectrum => {
            // "spectrum" - multiply by sampling frequency
            let fs_val_s: T = data.fs.as_();
            psd_avg.iter().map(|&p| p * fs_val_s).collect()
        }
    };

    Ok(WelchResult {
        frequencies: result_freqs,
        psd: result_psd,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32() {
        let arr = [
            -0.1408658, 2.9372524, -0.1190795, -1.7635250, -0.6057479, 1.4837575, -0.0584684,
            -1.4113780, 0.3412108, 1.9529803, 0.9757032, -1.6514435, 0.1201296, 1.2639116,
            -1.6568692, -1.3705078, -1.1910530, 1.6116732, 0.8686683, -1.0897907, -0.3721372,
            1.6204341, 0.0169285, -2.3782079, 0.4855662, 1.7210484, -1.4290131, -2.0420663,
            -0.1169624, 2.5862947, -0.3502908, -1.5385362, -0.3652317, 0.9226314, -0.0464172,
            -2.4029318, -0.5103204, 1.2521439, 1.7015659, -2.5694200, 0.2685705, 1.1861761,
            0.2991925, -2.6349077, -0.8252566, 0.8531086, -0.7525308, -3.4186978, 0.4530761,
            2.5846511, 0.1181391, -1.9967366, -0.2943062, 2.1522817, 1.0550684, -3.0852880,
            -0.9236702, 3.6964357, 1.2976173, -1.4785488, 0.2932672, 3.5134584, -0.5393023,
            -2.2890676, 1.3038109, 2.7086054, 0.6710210, -2.7763946, 0.0930381, 1.3763270,
            -0.0978434, -1.7785001, -0.4427855, 1.5855303, -0.6581872, -1.9885585, 0.5029719,
            1.9237372, -0.3134466, -1.3292285, -0.0581373, 1.4979491, 0.5212650, -1.9533608,
            -1.4334078, 1.5661753, 0.4821967, -1.2326114, -0.2093128, 2.2990309, -0.0174083,
            -2.1003210, 0.0496836, 2.6244476, -0.7921896, -2.0503481, -0.2421892, 1.6740196,
            -2.1924779, -1.5239072, -1.1549770, 1.6717711, 1.2384372, -2.3892707, 0.3226283,
            0.7167288, 0.3875305, -2.1897474, 0.0814763, 3.6614124, -0.8756540, -2.7575103,
            0.2446263, 1.3789440, -0.7474357, -0.7964077, 0.3368673, 1.0901363, 0.5361423,
            -2.0788218, -0.7260625, 0.4822067, 0.5243968, -0.8535628, -0.1023108, 2.5127417,
            0.8747677, -1.8722727,
        ];
        let q = welch_impl::<f32>(
            &Welch::new(&arr)
                .fs(4.0)
                .window(WelchWindow::Hann)
                .nperseg(128)
                .noverlap(64)
                .detrend(DetrendingMethod::Constant),
        )
        .unwrap();
        println!("{:?}", q.frequencies);
        println!("{:?}", q.psd);
    }

    #[test]
    fn test_f32_2() {
        let arr = [
            -0.1408658, 2.9372524, -0.1190795, -1.7635250, -0.6057479, 1.4837575, -0.0584684,
            -1.4113780, 0.3412108, 1.9529803, 0.9757032, -1.6514435, 0.1201296, 1.2639116,
            -1.6568692, -1.3705078, -1.1910530, 1.6116732, 0.8686683, -1.0897907, -0.3721372,
            1.6204341, 0.0169285, -2.3782079, 0.4855662, 1.7210484, -1.4290131, -2.0420663,
            -0.1169624, 2.5862947, -0.3502908, -1.5385362, -0.3652317, 0.9226314, -0.0464172,
            -2.4029318, -0.5103204, 1.2521439, 1.7015659, -2.5694200, 0.2685705, 1.1861761,
            0.2991925, -2.6349077, -0.8252566, 0.8531086, -0.7525308, -3.4186978, 0.4530761,
            2.5846511, 0.1181391, -1.9967366, -0.2943062, 2.1522817, 1.0550684, -3.0852880,
            -0.9236702, 3.6964357, 1.2976173, -1.4785488, 0.2932672, 3.5134584, -0.5393023,
            -2.2890676, 1.3038109, 2.7086054, 0.6710210, -2.7763946, 0.0930381, 1.3763270,
            -0.0978434, -1.7785001, -0.4427855, 1.5855303, -0.6581872, -1.9885585, 0.5029719,
            1.9237372, -0.3134466, -1.3292285, -0.0581373, 1.4979491, 0.5212650, -1.9533608,
            -1.4334078, 1.5661753, 0.4821967, -1.2326114, -0.2093128, 2.2990309, -0.0174083,
            -2.1003210, 0.0496836, 2.6244476, -0.7921896, -2.0503481, -0.2421892, 1.6740196,
            -2.1924779, -1.5239072, -1.1549770, 1.6717711, 1.2384372, -2.3892707, 0.3226283,
            0.7167288, 0.3875305, -2.1897474, 0.0814763, 3.6614124, -0.8756540, -2.7575103,
            0.2446263, 1.3789440, -0.7474357, -0.7964077, 0.3368673, 1.0901363, 0.5361423,
            -2.0788218, -0.7260625, 0.4822067, 0.5243968, -0.8535628, -0.1023108, 2.5127417,
            0.8747677, -1.8722727, -0.52321, 0.433121, -0.7260625,
        ];
        let q = welch_impl::<f32>(
            &Welch::new(&arr)
                .fs(4.0)
                .window(WelchWindow::Hann)
                .nperseg(128)
                .noverlap(64)
                .detrend(DetrendingMethod::Constant),
        )
        .unwrap();
        println!("{:?}", q.frequencies);
        println!("{:?}", q.psd);
    }
}
