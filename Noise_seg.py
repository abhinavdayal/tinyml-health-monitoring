
import numpy as np
import numpy
import pywt
import ewtpy

class Processed:
    def soft_threshold(x, threshold):
        """ 
        Soft thresholding function.
        """
    
        return np.sign(x) * np.maximum(0, np.abs(x) - threshold)

    def denoise_image_wavelet(image, wavelet='db1', level=1, threshold=0.1):
    
        coeffs = pywt.wavedec2(image, wavelet, level=level)
        coeffs_thresh = [coeffs[0]]
        for detail_level in range(1, len(coeffs)):
            coeff_arr = coeffs[detail_level]
            coeffs_thresh.append(tuple(Processed.soft_threshold(c, threshold) for c in coeff_arr))

    # Reconstruct the image from the denoised coefficients
        denoised_image = pywt.waverec2(coeffs_thresh, wavelet)
    
        return denoised_image



    def saewt_segmentation(signal, threshold=0.1):
        ewt, _, _ = ewtpy.EWT1D(signal, N=2)
        power_spectrum = np.abs(ewt)**2
        segmented_signals = []
        for i in range(power_spectrum.shape[1]):
            mask = power_spectrum[:, i] > threshold
            segmented_signal = ewt[:, i] * mask
            segmented_signals.append(segmented_signal)

        return np.array(segmented_signals)
