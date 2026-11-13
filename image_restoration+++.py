"""
Automatic Image Deblurring & Restoration Software
Single-file desktop application using traditional Digital Image Processing filters.

Install:
    pip install opencv-python numpy scipy pillow scikit-image

Run:
    python image_restoration.py

What's new in this version:
- The image is analysed immediately after upload (impulse noise, Gaussian noise,
  blur level / blur type, contrast and exposure).
- A restoration plan is recommended automatically: filter, kernel size,
  strength and the reason for each choice.
- The recommended plan is applied automatically after upload
  (can be switched off with the "Auto after upload" checkbox).
- A Recommendation Panel shows every selected step with its reason.
- Multiple filters: the recommended plan can contain several steps (e.g. impulse
  removal -> denoise -> deblur -> contrast). It is loaded into an editable
  Filter Pipeline where you can add, remove, reorder and run filters.
- Manual override: the controls on the right are filled with the recommended
  values, but you can change the filter and every parameter yourself.

Notes:
- All detection is heuristic, not ground truth. Review the result.
- Blind deblurring is an ill-posed problem. Blur type (Gaussian vs. motion)
  and motion angle are estimated from gradient statistics and may be wrong;
  use the manual controls to correct them.
- PSNR/SSIM are calculated only when a reference image is loaded.
"""

import os
import time
import threading
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.signal import wiener as scipy_wiener

try:
    from skimage.restoration import richardson_lucy
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


APP_TITLE = "Image Deblurring & Restoration Studio"
SUPPORTED = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

FILTER_NAMES = [
    "Mean", "Median", "Adaptive Median", "Gaussian", "Bilateral",
    "Wiener Denoise", "Non-Local Means", "Wiener Deconvolution",
    "Richardson-Lucy", "Laplacian Sharpen", "Unsharp Masking",
    "High-Boost", "CLAHE", "Gamma Correction",
]
# Which parameters each filter actually uses (used for display and tooltips).
KERNEL_FILTERS = {"Mean", "Median", "Adaptive Median", "Gaussian", "Bilateral",
                  "Wiener Denoise", "Non-Local Means", "Wiener Deconvolution",
                  "Richardson-Lucy", "High-Boost"}
STRENGTH_FILTERS = {"Gaussian", "Bilateral", "Non-Local Means",
                    "Wiener Deconvolution", "Richardson-Lucy",
                    "Laplacian Sharpen", "Unsharp Masking", "High-Boost",
                    "CLAHE", "Gamma Correction"}


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def read_image(path):
    """Read an image robustly, including paths containing non-ASCII characters."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not open this image.")
    return image


def save_image(path, image):
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        ext = ".png"
        path += ext
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("Could not encode the output image.")
    encoded.tofile(path)


def odd_kernel(value, minimum=3, maximum=51):
    value = max(minimum, min(maximum, int(value)))
    return value if value % 2 else value + 1


def clamp(value, low, high):
    return max(low, min(high, value))


def to_display_rgb(bgr, max_size=(560, 430)):
    h, w = bgr.shape[:2]
    if max_size is None:  # actual size (preview panes scroll instead)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scale = min(max_size[0] / max(w, 1), max_size[1] / max(h, 1), 1.0)
    if scale < 1:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Detection and quality analysis
# ---------------------------------------------------------------------------

def detect_impulses(image):
    """Find salt-and-pepper-like pixels.

    A pixel counts as an impulse when, in at least one channel, it is almost
    black or almost white AND differs strongly from its 3x3 median. This avoids
    flagging legitimately saturated regions (white sky, black backgrounds).
    Returns (per-channel mask, median-filtered image).
    """
    med = cv2.medianBlur(image, 3)
    diff = cv2.absdiff(image, med)
    extreme = (image <= 5) | (image >= 250)
    mask = extreme & (diff > 50)
    if mask.ndim == 2:
        mask = mask[:, :, None]
    return mask, med


def estimate_gaussian_sigma(gray):
    """Robust Gaussian-noise sigma (Immerkaer's Laplacian-difference mask,
    with a median-based statistic so edges/texture bias it less)."""
    if min(gray.shape[:2]) < 5:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    response = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
    response = response[1:-1, 1:-1]
    # std of the response is 6*sigma; median(|x|) = 0.6745*std for a Gaussian.
    return float(np.median(np.abs(response)) / (0.6745 * 6.0))


def estimate_motion_direction(gray):
    """Estimate dominant gradient anisotropy and the likely motion-blur angle.

    Motion blur suppresses gradients along the blur direction, so the blur
    direction is perpendicular to the dominant gradient orientation.
    Returns (anisotropy 0..1, blur angle in degrees 0..180).
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float(np.sum(gx * gx))
    jyy = float(np.sum(gy * gy))
    jxy = float(np.sum(gx * gy))
    total = jxx + jyy
    if total < 1e-6:
        return 0.0, 0
    anisotropy = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / total)
    grad_angle = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
    blur_angle = (grad_angle + 90.0) % 180.0
    return anisotropy, int(round(blur_angle / 5.0) * 5) % 180


def analyze_image(image):
    """Analyse noise, blur, contrast and exposure. Heuristic, not ground truth."""
    mask, med = detect_impulses(image)
    impulse_ratio = float(mask.any(axis=2).mean())

    # Remove impulses before measuring everything else so they do not
    # pollute the Gaussian-noise and sharpness estimates.
    clean = np.where(mask, med, image).astype(np.uint8)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    # Per-channel estimate: noise is independent per channel, so measuring on
    # grayscale would under-report it by roughly a third.
    sigma = float(np.mean([estimate_gaussian_sigma(c) for c in cv2.split(clean)]))

    # Sharpness is measured on a size-normalised copy so the thresholds
    # behave similarly for small and large images.
    h, w = gray.shape
    scale = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(gray, (max(8, int(w * scale)), max(8, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else gray
    sigma_small = estimate_gaussian_sigma(small)
    lap_var = float(cv2.Laplacian(small, cv2.CV_64F).var())
    # White noise adds 20*sigma^2 to the variance of the Laplacian response.
    sharpness = max(0.0, lap_var - 20.0 * sigma_small ** 2)

    if sharpness < 20:
        blur_level = "very strong"
    elif sharpness < 45:
        blur_level = "strong"
    elif sharpness < 120:
        blur_level = "moderate"
    else:
        blur_level = "none"

    anisotropy, motion_angle = estimate_motion_direction(small)
    motion_likely = anisotropy > 0.30 and blur_level in ("strong", "very strong")

    contrast = float(gray.std())
    mean_intensity = float(gray.mean())
    contrast_label = "Low contrast" if contrast < 35 else "Normal contrast"
    if mean_intensity < 65:
        exposure = "Dark"
    elif mean_intensity > 195:
        exposure = "Bright"
    else:
        exposure = "Normal exposure"

    has_impulse = impulse_ratio >= 0.002
    has_gauss = sigma >= 2.5
    if has_impulse and has_gauss:
        noise_type = "Mixed (impulse + Gaussian-like)"
    elif has_impulse:
        noise_type = "Impulse (salt-and-pepper-like)"
    elif has_gauss:
        noise_type = "Gaussian-like"
    else:
        noise_type = "None / very low"

    return {
        "impulse_ratio": impulse_ratio,
        "gaussian_sigma": sigma,
        "noise_type": noise_type,
        "sharpness": sharpness,
        "blur_level": blur_level,
        "anisotropy": anisotropy,
        "motion_angle": motion_angle,
        "motion_likely": motion_likely,
        "contrast_std": contrast,
        "contrast_label": contrast_label,
        "mean_intensity": mean_intensity,
        "exposure": exposure,
        "size": (w, h),
    }


def format_analysis(info):
    blur = info["blur_level"]
    if blur == "none":
        blur_text = "no strong blur detected"
    elif info["motion_likely"]:
        blur_text = f"{blur} blur, looks like motion (~{info['motion_angle']} deg)"
    else:
        blur_text = f"{blur} blur, looks like defocus/Gaussian"
    return (
        "AUTOMATIC IMAGE ANALYSIS\n"
        f"Noise type       : {info['noise_type']}\n"
        f"Impulse pixels   : {info['impulse_ratio'] * 100:.2f}%\n"
        f"Gaussian sigma   : {info['gaussian_sigma']:.2f}\n"
        f"Blur             : {blur_text}\n"
        f"Sharpness score  : {info['sharpness']:.1f}\n"
        f"Contrast         : {info['contrast_label']} (std {info['contrast_std']:.1f})\n"
        f"Exposure         : {info['exposure']} (mean {info['mean_intensity']:.0f})\n"
    )


def quality_metrics(reference, restored):
    if reference.shape[:2] != restored.shape[:2]:
        restored = cv2.resize(restored, (reference.shape[1], reference.shape[0]))
    mse = float(np.mean((reference.astype(np.float32) -
                         restored.astype(np.float32)) ** 2))
    if SKIMAGE_AVAILABLE:
        ref_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
        out_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
        psnr = peak_signal_noise_ratio(ref_rgb, out_rgb, data_range=255)
        ssim = structural_similarity(ref_rgb, out_rgb, channel_axis=2,
                                     data_range=255)
        return f"PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}   MSE: {mse:.2f}"
    return f"MSE: {mse:.2f} (install scikit-image for PSNR/SSIM)"


# ---------------------------------------------------------------------------
# Traditional DIP filters
# ---------------------------------------------------------------------------

def mean_filter(image, k=3):
    return cv2.blur(image, (k, k))


def median_filter(image, k=3):
    return cv2.medianBlur(image, odd_kernel(k, 3, 51))


def adaptive_median_filter(image, max_k=7):
    """Adaptive median filter (vectorised), applied channel-wise.

    Window grows from 3x3 up to max_k x max_k. Same decision logic as the
    classic algorithm (stage A / stage B) but computed for whole images with
    OpenCV min/max/median filters, so it is fast enough for auto-processing.
    """
    max_k = odd_kernel(max_k, 3, 15)
    channels = [image] if image.ndim == 2 else cv2.split(image)
    outputs = []
    for src in channels:
        output = src.copy()
        unresolved = np.ones(src.shape, dtype=bool)
        zmed = src
        for size in range(3, max_k + 1, 2):
            kern = np.ones((size, size), np.uint8)
            zmin = cv2.erode(src, kern, borderType=cv2.BORDER_REFLECT)
            zmax = cv2.dilate(src, kern, borderType=cv2.BORDER_REFLECT)
            zmed = cv2.medianBlur(src, size)
            stage_a = unresolved & (zmin < zmed) & (zmed < zmax)
            keep = (zmin < src) & (src < zmax)
            use_src = stage_a & keep
            use_med = stage_a & ~keep
            output[use_src] = src[use_src]
            output[use_med] = zmed[use_med]
            unresolved &= ~stage_a
        output[unresolved] = zmed[unresolved]
        outputs.append(output)
    return outputs[0] if image.ndim == 2 else cv2.merge(outputs)


def gaussian_filter(image, k=5, sigma=1.2):
    return cv2.GaussianBlur(image, (odd_kernel(k), odd_kernel(k)), sigma)


def bilateral_filter(image, diameter=9, sigma_color=60, sigma_space=60):
    return cv2.bilateralFilter(image, int(diameter), sigma_color, sigma_space)


def wiener_denoise(image, k=5):
    """Local Wiener denoising, channel by channel."""
    channels = [image] if image.ndim == 2 else cv2.split(image)
    result = []
    for ch in channels:
        arr = scipy_wiener(ch.astype(np.float32), (k, k))
        result.append(np.uint8(np.clip(np.nan_to_num(arr), 0, 255)))
    return result[0] if image.ndim == 2 else cv2.merge(result)


def nlm_denoise(image, strength=10, template=7):
    template = odd_kernel(template, 3, 11)
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength,
                                           template, 21)


def laplacian_sharpen(image, amount=0.7):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    lap = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    return cv2.addWeighted(image, 1.0, lap, -amount, 0)


def unsharp_mask(image, amount=1.2, sigma=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def high_boost(image, amount=1.0, k=5):
    k = odd_kernel(k, 3, 51)
    blurred = cv2.GaussianBlur(image, (k, k), 0)
    mask = cv2.subtract(image, blurred)
    return cv2.addWeighted(image, 1.0, mask, amount, 0)


def clahe_enhance(image, clip=2.0, tile=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    merged = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_correct(image, gamma=1.15):
    inv = 1.0 / max(gamma, 0.01)
    table = np.array([((i / 255.0) ** inv) * 255
                      for i in range(256)]).clip(0, 255).astype("uint8")
    return cv2.LUT(image, table)


# ---------------------------------------------------------------------------
# Deconvolution
# ---------------------------------------------------------------------------

def build_psf(kind="Gaussian", size=9, angle=0):
    """Build a normalized point-spread function (PSF)."""
    size = odd_kernel(size, 3, 51)
    if kind == "Motion":
        psf = np.zeros((size, size), np.float32)
        center = size // 2
        length = max(3, size - 2)
        angle_rad = np.deg2rad(angle)
        dx = int(round(np.cos(angle_rad) * length / 2))
        dy = int(round(np.sin(angle_rad) * length / 2))
        cv2.line(psf, (center - dx, center - dy),
                 (center + dx, center + dy), 1, 1)
    else:
        psf = cv2.getGaussianKernel(size, max(size / 6.0, 0.5))
        psf = psf @ psf.T
    total = float(psf.sum())
    return psf / total if total else psf


def wiener_deconvolution(image, psf_kind="Gaussian", psf_size=9,
                         angle=0, balance=0.015):
    """Frequency-domain Wiener deconvolution using an assumed PSF.

    The image is reflect-padded first to reduce wrap-around ringing at borders.
    """
    psf = build_psf(psf_kind, psf_size, angle)
    ph, pw = psf.shape
    pad = max(ph, pw)
    padded_img = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                    cv2.BORDER_REFLECT)
    h, w = padded_img.shape[:2]
    kernel = np.zeros((h, w), np.float32)
    kernel[:ph, :pw] = psf
    kernel = np.roll(kernel, (-(ph // 2), -(pw // 2)), axis=(0, 1))
    H = np.fft.fft2(kernel)
    output = []
    for channel in cv2.split(padded_img):
        G = np.fft.fft2(channel.astype(np.float32))
        F = (np.conj(H) / (np.abs(H) ** 2 + balance)) * G
        restored = np.fft.ifft2(F).real[pad:-pad, pad:-pad]
        output.append(np.uint8(np.clip(restored, 0, 255)))
    return cv2.merge(output)


def richardson_lucy_deblur(image, psf_kind="Gaussian", psf_size=9,
                           iterations=20, angle=0):
    if not SKIMAGE_AVAILABLE:
        raise RuntimeError("Richardson-Lucy requires scikit-image. Install it with pip.")
    psf = build_psf(psf_kind, psf_size, angle)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    result = np.zeros_like(rgb)
    for c in range(3):
        result[:, :, c] = richardson_lucy(rgb[:, :, c], psf,
                                          num_iter=int(iterations),
                                          clip=False)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Filter dispatch (shared by manual and automatic modes)
# ---------------------------------------------------------------------------

def apply_filter(img, name, k=5, strength=1.0, psf="Gaussian", angle=0):
    """Apply one named filter with the unified (kernel, strength) parameters."""
    k = odd_kernel(k)
    if name == "Mean":
        return mean_filter(img, k)
    if name == "Median":
        return median_filter(img, k)
    if name == "Adaptive Median":
        return adaptive_median_filter(img, min(k, 15))
    if name == "Gaussian":
        return gaussian_filter(img, k, strength)
    if name == "Bilateral":
        return bilateral_filter(img, max(5, k), 50 * strength, 50 * strength)
    if name == "Wiener Denoise":
        return wiener_denoise(img, k)
    if name == "Non-Local Means":
        return nlm_denoise(img, max(1, int(round(10 * strength))), k)
    if name == "Wiener Deconvolution":
        return wiener_deconvolution(img, psf, k, angle,
                                    max(0.001, 0.03 / strength))
    if name == "Richardson-Lucy":
        return richardson_lucy_deblur(img, psf, k,
                                      max(5, int(20 * strength)), angle)
    if name == "Laplacian Sharpen":
        return laplacian_sharpen(img, strength)
    if name == "Unsharp Masking":
        return unsharp_mask(img, strength, max(0.3, strength))
    if name == "High-Boost":
        return high_boost(img, strength, k)
    if name == "CLAHE":
        return clahe_enhance(img, max(1.0, strength * 2))
    if name == "Gamma Correction":
        return gamma_correct(img, max(0.2, strength))
    raise ValueError(f"Unknown filter: {name}")


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

@dataclass
class Step:
    filter: str
    kernel: int = 5
    strength: float = 1.0
    reason: str = ""
    psf: str = "Gaussian"
    angle: int = 0

    def kernel_text(self):
        if self.filter not in KERNEL_FILTERS:
            return "n/a (not used by this filter)"
        if self.filter == "Adaptive Median":
            return f"{self.kernel} x {self.kernel} (maximum window)"
        if self.filter in ("Wiener Deconvolution", "Richardson-Lucy"):
            return f"{self.kernel} x {self.kernel} ({self.psf} PSF size)"
        return f"{self.kernel} x {self.kernel}"

    def strength_text(self):
        if self.filter not in STRENGTH_FILTERS:
            return "n/a (not used by this filter)"
        return f"{self.strength:.2f}"


def recommend_plan(info):
    """Turn an analysis dict into an ordered list of restoration steps.

    Order matters: impulse removal -> Gaussian denoising -> deblurring ->
    contrast/exposure. Denoising comes before deblurring because deconvolution
    amplifies noise.
    """
    steps = []
    ratio = info["impulse_ratio"]
    sigma = info["gaussian_sigma"]

    # 1) Impulse noise
    if ratio >= 0.01:
        max_k = 5 if ratio < 0.03 else 7 if ratio < 0.08 else 9
        steps.append(Step(
            "Adaptive Median", max_k, 1.0,
            f"{ratio * 100:.1f}% of pixels look like salt-and-pepper impulses. "
            f"Adaptive median removes them while keeping edges; the window "
            f"grows up to {max_k}x{max_k} because the impulse density is "
            f"{'moderate' if ratio < 0.03 else 'high' if ratio < 0.08 else 'very high'}."))
    elif ratio >= 0.002:
        steps.append(Step(
            "Median", 3, 1.0,
            f"A few impulse pixels ({ratio * 100:.2f}%) detected. A 3x3 median "
            f"removes them with minimal blurring."))

    # 2) Gaussian-like noise
    if sigma >= 2.5:
        if sigma < 6:
            steps.append(Step(
                "Bilateral", 5, round(clamp(sigma * 3 / 50, 0.3, 3.0), 2),
                f"Low Gaussian-like noise (sigma ~ {sigma:.1f}). Bilateral "
                f"filtering smooths noise while preserving edges; strength is "
                f"scaled to the noise level."))
        else:
            k = 5 if sigma < 12 else 7 if sigma < 25 else 9
            steps.append(Step(
                "Non-Local Means", k, round(clamp(sigma * 1.05 / 10, 0.3, 3.0), 2),
                f"{'Moderate' if sigma < 12 else 'High'} Gaussian-like noise "
                f"(sigma ~ {sigma:.1f}). Non-Local Means preserves texture "
                f"best; template window {k}x{k} and strength follow the noise level."))

    # 3) Blur
    level = info["blur_level"]
    noisy = sigma >= 6
    if level in ("strong", "very strong"):
        very = level == "very strong"
        if info["motion_likely"]:
            length = 17 if very else 11
            steps.append(Step(
                "Wiener Deconvolution", length + 2, 1.2 if not noisy else 0.7,
                f"{level.capitalize()} blur with strongly directional gradients "
                f"(anisotropy {info['anisotropy']:.2f}), which suggests motion "
                f"blur at ~{info['motion_angle']} deg. Wiener deconvolution with a "
                f"motion PSF (length ~{length}px) is used"
                f"{'; strength lowered to limit noise amplification' if noisy else ''}.",
                psf="Motion", angle=info["motion_angle"]))
        else:
            k = 15 if very else 11
            strength = 2.0 if very else 1.5
            if noisy:
                strength = 0.7  # more regularisation: deconvolution amplifies noise
            steps.append(Step(
                "Wiener Deconvolution", k, strength,
                f"{level.capitalize()} blur without a dominant direction, so a "
                f"Gaussian (defocus-like) PSF of size {k} is assumed. Wiener "
                f"deconvolution is regularised and stable; Richardson-Lucy "
                f"(available manually) rings badly when the PSF guess is off"
                f"{'. Strength is lowered because the image is noisy' if noisy else ''}."))
    elif level == "moderate":
        strength = 1.0 if info["sharpness"] < 80 else 0.8
        steps.append(Step(
            "Unsharp Masking", 5, strength,
            f"Moderate blur (sharpness score {info['sharpness']:.0f}). Unsharp "
            f"masking restores edge contrast without the noise risk of full deconvolution."))

    # 4) Contrast / exposure
    if info["contrast_label"] == "Low contrast":
        std = info["contrast_std"]
        steps.append(Step(
            "CLAHE", 5, 1.0 if std >= 25 else 1.5,
            f"Low global contrast (std {std:.1f}). CLAHE stretches contrast "
            f"locally; it runs last so it does not amplify noise."))
    elif info["exposure"] == "Dark":
        steps.append(Step(
            "Gamma Correction", 5, 1.35,
            f"The image is dark (mean {info['mean_intensity']:.0f}). Gamma > 1 "
            f"brightens shadows."))
    elif info["exposure"] == "Bright":
        steps.append(Step(
            "Gamma Correction", 5, 0.75,
            f"The image is very bright (mean {info['mean_intensity']:.0f}). "
            f"Gamma < 1 recovers highlights."))
    return steps


def apply_plan(image, plan):
    result = image.copy()
    for step in plan:
        result = apply_filter(result, step.filter, step.kernel, step.strength,
                              step.psf, step.angle)
    return result


def format_recommendation(plan, info=None):
    if not plan:
        text = ("No restoration recommended.\n"
                "No significant noise, blur, contrast or exposure problem was "
                "detected, so the original is kept.\n\n")
        if info is not None:
            text += (
                "Measured values for THIS image (trigger level in brackets):\n"
                f"  Impulse pixels : {info['impulse_ratio'] * 100:.2f}%  [>= 0.20%]\n"
                f"  Gaussian sigma : {info['gaussian_sigma']:.2f}  [>= 2.5]\n"
                f"  Sharpness      : {info['sharpness']:.1f}  [< 120 = blur]\n"
                f"  Contrast (std) : {info['contrast_std']:.1f}  [< 35 = low]\n"
                f"  Mean intensity : {info['mean_intensity']:.0f}  [< 65 dark, > 195 bright]\n\n")
        return text + "You can still pick any filter manually on the right."
    lines = []
    for i, step in enumerate(plan, 1):
        lines.append(f"STEP {i}: {step.filter}")
        lines.append(f"  Kernel   : {step.kernel_text()}")
        lines.append(f"  Strength : {step.strength_text()}")
        if step.filter in ("Wiener Deconvolution", "Richardson-Lucy"):
            extra = f"{step.psf} PSF"
            if step.psf == "Motion":
                extra += f", angle {step.angle} deg"
            lines.append(f"  PSF      : {extra}")
        lines.append(f"  Reason   : {step.reason}")
        lines.append("")
    lines.append("The whole plan is loaded into the Filter Pipeline tab (editable). "
                 "Step 1 is loaded into the manual controls; change any value "
                 "and press 'Apply Selected Filter' for a single filter, or "
                 "edit/run the pipeline for several.")
    return "\n".join(lines)


def automatic_restore(image):
    """Analyse + recommend + apply. Returns (result, info, plan)."""
    info = analyze_image(image)
    plan = recommend_plan(info)
    return apply_plan(image, plan), info, plan


# ---------------------------------------------------------------------------
# Tkinter application
# ---------------------------------------------------------------------------

class ImageRestorationApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1320x900")
        self.root.minsize(1060, 720)

        self.original = None
        self.restored = None
        self.reference = None
        self.input_path = None
        self.original_photo = None
        self.restored_photo = None
        self.processing = False
        self.current_plan = []
        self.pipeline = []
        self._analysis_base = ""

        self._configure_style()
        self._build_ui()

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=7, font=("Segoe UI", 9))
        style.configure("Accent.TButton", padding=8, font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#555555")
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))

    def _make_text_panel(self, parent, title, height):
        box = ttk.LabelFrame(parent, text=title, padding=6)
        text = tk.Text(box, height=height, wrap="word",
                       font=("Consolas", 9), state="disabled")
        scroll = ttk.Scrollbar(box, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        return box, text

    def _make_scroll_preview(self, parent, placeholder):
        """Preview area that shows scrollbars only when the image is larger
        than the available space. Returns (container, label)."""
        holder = ttk.Frame(parent)
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)
        canvas = tk.Canvas(holder, highlightthickness=0, bd=0)
        vbar = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        hbar = ttk.Scrollbar(holder, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        label = ttk.Label(canvas, text=placeholder, anchor="center")
        win = canvas.create_window(0, 0, window=label, anchor="nw")

        def layout(_event=None):
            total_w, total_h = holder.winfo_width(), holder.winfo_height()
            lw, lh = label.winfo_reqwidth(), label.winfo_reqheight()
            if total_w <= 1 or total_h <= 1:
                return
            bar_w, bar_h = vbar.winfo_reqwidth(), hbar.winfo_reqheight()
            need_v = lh > total_h
            need_h = lw > total_w
            if need_h and not need_v:
                need_v = lh > total_h - bar_h
            if need_v and not need_h:
                need_h = lw > total_w - bar_w
            if need_v:
                vbar.grid(row=0, column=1, sticky="ns")
            else:
                vbar.grid_remove()
            if need_h:
                hbar.grid(row=1, column=0, sticky="ew")
            else:
                hbar.grid_remove()
            view_w = total_w - (bar_w if need_v else 0)
            view_h = total_h - (bar_h if need_h else 0)
            width, height = max(lw, view_w), max(lh, view_h)
            canvas.itemconfigure(win, width=width, height=height)
            canvas.configure(scrollregion=(0, 0, width, height))

        def on_wheel(event):
            if event.num == 4 or event.delta > 0:
                step = -1
            else:
                step = 1
            if event.state & 0x0001:  # Shift = horizontal
                canvas.xview_scroll(step * 3, "units")
            else:
                canvas.yview_scroll(step * 3, "units")

        def bind_wheel(_event=None):
            canvas.bind_all("<MouseWheel>", on_wheel)
            canvas.bind_all("<Button-4>", on_wheel)
            canvas.bind_all("<Button-5>", on_wheel)

        def unbind_wheel(_event=None):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        holder.bind("<Configure>", layout)
        label.bind("<Configure>", layout)
        holder.bind("<Enter>", bind_wheel)
        holder.bind("<Leave>", unbind_wheel)
        return holder, label

    def _build_ui(self):
        header = ttk.Frame(self.root, padding=(14, 10))
        header.pack(fill="x")
        ttk.Label(header, text="Image Deblurring & Restoration Studio",
                  style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Traditional Digital Image Processing",
                  style="Sub.TLabel").pack(side="right", pady=8)

        toolbar = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Upload Image", command=self.upload_image,
                   style="Accent.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Load Reference", command=self.load_reference).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Automatic Restore", command=self.run_automatic,
                   style="Accent.TButton").pack(side="left", padx=4)
        ttk.Button(toolbar, text="Save Result", command=self.save_result).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side="left", padx=4)

        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Auto after upload",
                        variable=self.auto_var).pack(side="left", padx=12)

        self.actual_size_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="Actual size (scroll)",
                        variable=self.actual_size_var,
                        command=self._refresh_previews).pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Upload an image to begin.")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="right")

        main = ttk.Panedwindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=12, pady=6)

        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=3)
        main.add(right, weight=1)

        previews = ttk.Frame(left)
        previews.pack(fill="both", expand=True)
        original_box = ttk.LabelFrame(previews, text="Original Image", padding=8)
        restored_box = ttk.LabelFrame(previews, text="Restored Image", padding=8)
        original_box.pack(side="left", fill="both", expand=True, padx=(0, 5))
        restored_box.pack(side="left", fill="both", expand=True, padx=(5, 0))

        original_view, self.original_label = self._make_scroll_preview(
            original_box, "No image loaded")
        original_view.pack(fill="both", expand=True)
        restored_view, self.restored_label = self._make_scroll_preview(
            restored_box, "Restored result appears here")
        restored_view.pack(fill="both", expand=True)

        lower = ttk.Frame(left)
        lower.pack(fill="x", pady=(8, 0))
        lower.columnconfigure(0, weight=1, uniform="panels")
        lower.columnconfigure(1, weight=1, uniform="panels")
        analysis_box, self.analysis_text = self._make_text_panel(
            lower, "Image Analysis", 11)
        recommend_box, self.recommend_text = self._make_text_panel(
            lower, "Recommendation Panel", 11)
        analysis_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        recommend_box.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self._set_text(self.recommend_text,
                       "The recommended filter, kernel, strength and reason "
                       "will appear here after an image is analysed.")

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)
        controls = ttk.Frame(notebook, padding=10)
        pipe_tab = ttk.Frame(notebook, padding=10)
        test_tab = ttk.Frame(notebook, padding=10)
        notebook.add(controls, text="Manual Override")
        notebook.add(pipe_tab, text="Filter Pipeline")
        notebook.add(test_tab, text="Test Images")
        self.notebook = notebook

        ttk.Label(controls, text="Filter").pack(anchor="w")
        self.filter_var = tk.StringVar(value="Median")
        ttk.Combobox(controls, textvariable=self.filter_var,
                     values=FILTER_NAMES, state="readonly").pack(fill="x", pady=(3, 8))

        ttk.Label(controls, text="Kernel size").pack(anchor="w")
        self.kernel_var = tk.IntVar(value=5)
        ttk.Spinbox(controls, from_=3, to=31, increment=2,
                    textvariable=self.kernel_var, width=10).pack(anchor="w", pady=(3, 8))

        strength_row = ttk.Frame(controls)
        strength_row.pack(fill="x")
        ttk.Label(strength_row, text="Strength / sigma").pack(side="left")
        self.strength_var = tk.DoubleVar(value=1.2)
        self.strength_text_var = tk.StringVar(value="1.20")
        ttk.Label(strength_row, textvariable=self.strength_text_var).pack(side="right")
        self.strength_var.trace_add(
            "write", lambda *_: self.strength_text_var.set(
                f"{self._safe_float(self.strength_var, 1.0):.2f}"))
        ttk.Scale(controls, from_=0.1, to=3.0, variable=self.strength_var,
                  orient="horizontal").pack(fill="x", pady=(3, 8))

        ttk.Label(controls, text="Deconvolution PSF").pack(anchor="w")
        self.psf_var = tk.StringVar(value="Gaussian")
        ttk.Combobox(controls, textvariable=self.psf_var,
                     values=["Gaussian", "Motion"], state="readonly").pack(fill="x", pady=(3, 8))

        ttk.Label(controls, text="Motion angle (degrees)").pack(anchor="w")
        self.angle_var = tk.IntVar(value=0)
        ttk.Spinbox(controls, from_=0, to=180, increment=5,
                    textvariable=self.angle_var, width=10).pack(anchor="w", pady=(3, 8))

        self.chain_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Apply on current result (chain filters)",
                        variable=self.chain_var).pack(anchor="w", pady=(0, 6))

        ttk.Button(controls, text="Apply Selected Filter",
                   command=self.run_manual, style="Accent.TButton").pack(fill="x", pady=(2, 4))
        ttk.Button(controls, text="Add to Pipeline  \u2192",
                   command=self.pipeline_add).pack(fill="x", pady=3)
        ttk.Button(controls, text="Use Recommended Settings",
                   command=self.use_recommended).pack(fill="x", pady=3)
        ttk.Button(controls, text="Use Restored as Original",
                   command=self.promote_result).pack(fill="x", pady=3)

        # --- Filter pipeline tab: several filters, run top to bottom ---
        ttk.Label(pipe_tab, text="Filters run from top to bottom:").pack(anchor="w")
        list_frame = ttk.Frame(pipe_tab)
        list_frame.pack(fill="both", expand=True, pady=(4, 6))
        self.pipeline_list = tk.Listbox(list_frame, height=10, exportselection=False,
                                        font=("Consolas", 9), activestyle="dotbox")
        pscroll = ttk.Scrollbar(list_frame, orient="vertical",
                                command=self.pipeline_list.yview)
        self.pipeline_list.configure(yscrollcommand=pscroll.set)
        pscroll.pack(side="right", fill="y")
        self.pipeline_list.pack(side="left", fill="both", expand=True)
        self.pipeline_list.bind("<<ListboxSelect>>", self.pipeline_select)

        pgrid = ttk.Frame(pipe_tab)
        pgrid.pack(fill="x")
        pgrid.columnconfigure(0, weight=1)
        pgrid.columnconfigure(1, weight=1)
        pbuttons = [("Move Up", lambda: self.pipeline_move(-1)),
                    ("Move Down", lambda: self.pipeline_move(1)),
                    ("Replace with Current", self.pipeline_replace),
                    ("Remove", self.pipeline_remove),
                    ("Load Recommended", self.pipeline_load_recommended),
                    ("Clear", self.pipeline_clear)]
        for i, (label, cmd) in enumerate(pbuttons):
            ttk.Button(pgrid, text=label, command=cmd).grid(
                row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        ttk.Button(pipe_tab, text="Run Pipeline", command=self.run_pipeline,
                   style="Accent.TButton").pack(fill="x", pady=(8, 4))
        ttk.Label(pipe_tab, wraplength=230, style="Sub.TLabel",
                  text="Select an entry to load its settings into the Manual "
                       "Override tab. Edit them, then 'Replace with Current'. "
                       "Order matters: denoise first, deblur next, "
                       "contrast last.").pack(anchor="w", pady=(4, 0))

        # --- Test images tab ---
        ttk.Label(test_tab, text="Add synthetic degradation to the working image:",
                  wraplength=230).pack(anchor="w", pady=(0, 6))
        tests = [("Add Gaussian Noise", "gaussian"), ("Add Salt & Pepper Noise", "sp"),
                 ("Add Gaussian Blur", "blur"), ("Add Motion Blur", "motion")]
        for label, kind in tests:
            ttk.Button(test_tab, text=label,
                       command=lambda k=kind: self.add_test_degradation(k)
                       ).pack(fill="x", pady=3)

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill="x", padx=12, pady=(0, 10))

    # -- small helpers ------------------------------------------------------

    @staticmethod
    def _safe_float(var, default):
        try:
            return float(var.get())
        except (tk.TclError, ValueError):
            return default

    @staticmethod
    def _safe_int(var, default):
        try:
            return int(var.get())
        except (tk.TclError, ValueError):
            return default

    def _set_status(self, message):
        self.status_var.set(message)

    @staticmethod
    def _set_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _show_analysis(self, text):
        self._analysis_base = text
        self._set_text(self.analysis_text, text)

    def _render(self, image, target):
        if image is None:
            target.configure(image="", text="No image")
            return None
        rgb = to_display_rgb(
            image, None if self.actual_size_var.get() else (560, 430))
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        target.configure(image=photo, text="")
        target.image = photo
        return photo

    def _refresh_previews(self):
        if self.original is not None:
            self.original_photo = self._render(self.original, self.original_label)
        if self.restored is not None:
            self.restored_photo = self._render(self.restored, self.restored_label)

    # -- recommendation panel / controls -------------------------------------

    def _show_recommendation(self, plan, info=None):
        self.current_plan = plan
        self._set_text(self.recommend_text, format_recommendation(plan, info))

    def _load_step_into_controls(self, step):
        self.filter_var.set(step.filter)
        self.kernel_var.set(step.kernel)
        self.strength_var.set(step.strength)
        self.psf_var.set(step.psf)
        self.angle_var.set(step.angle)

    def use_recommended(self):
        if not self.current_plan:
            messagebox.showinfo("No recommendation",
                                "There is no recommended filter for this image yet.")
            return
        self._load_step_into_controls(self.current_plan[0])
        self._set_status("Recommended settings loaded into the controls.")

    # -- multi-filter pipeline -----------------------------------------------

    def _current_step(self):
        return Step(self.filter_var.get(),
                    odd_kernel(self._safe_int(self.kernel_var, 5)),
                    round(self._safe_float(self.strength_var, 1.0), 2),
                    "Added manually", self.psf_var.get(),
                    self._safe_int(self.angle_var, 0))

    @staticmethod
    def _step_label(step):
        parts = [step.filter]
        if step.filter in KERNEL_FILTERS:
            parts.append(f"k={step.kernel}")
        if step.filter in STRENGTH_FILTERS:
            parts.append(f"s={step.strength:.2f}")
        if step.filter in ("Wiener Deconvolution", "Richardson-Lucy"):
            parts.append(step.psf + (f" {step.angle}\u00b0" if step.psf == "Motion" else ""))
        return " | ".join(parts)

    def _refresh_pipeline_list(self, select=None):
        self.pipeline_list.delete(0, "end")
        for i, step in enumerate(self.pipeline, 1):
            self.pipeline_list.insert("end", f"{i}. {self._step_label(step)}")
        if select is not None and 0 <= select < len(self.pipeline):
            self.pipeline_list.selection_set(select)
            self.pipeline_list.see(select)

    def _selected_index(self):
        sel = self.pipeline_list.curselection()
        return sel[0] if sel else None

    def pipeline_add(self):
        self.pipeline.append(self._current_step())
        self._refresh_pipeline_list(len(self.pipeline) - 1)
        self._set_status(f"Added to pipeline ({len(self.pipeline)} filters).")

    def pipeline_replace(self):
        i = self._selected_index()
        if i is None:
            messagebox.showinfo("Pipeline", "Select a pipeline entry to replace.")
            return
        self.pipeline[i] = self._current_step()
        self._refresh_pipeline_list(i)

    def pipeline_remove(self):
        i = self._selected_index()
        if i is None:
            return
        del self.pipeline[i]
        self._refresh_pipeline_list(min(i, len(self.pipeline) - 1))

    def pipeline_move(self, delta):
        i = self._selected_index()
        if i is None:
            return
        j = i + delta
        if 0 <= j < len(self.pipeline):
            self.pipeline[i], self.pipeline[j] = self.pipeline[j], self.pipeline[i]
            self._refresh_pipeline_list(j)

    def pipeline_clear(self):
        self.pipeline = []
        self._refresh_pipeline_list()

    def pipeline_load_recommended(self):
        if not self.current_plan:
            messagebox.showinfo("Pipeline", "There is no recommended plan for this image yet.")
            return
        self.pipeline = [Step(**vars(s)) for s in self.current_plan]
        self._refresh_pipeline_list(0)

    def pipeline_select(self, _event=None):
        i = self._selected_index()
        if i is not None:
            self._load_step_into_controls(self.pipeline[i])

    def run_pipeline(self):
        if self.original is None:
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        if not self.pipeline:
            messagebox.showinfo("Pipeline", "The pipeline is empty. Add filters first.")
            return
        steps = [Step(**vars(s)) for s in self.pipeline]
        chain = self.chain_var.get() and self.restored is not None
        source = (self.restored if chain else self.original).copy()

        def task():
            out = apply_plan(source, steps)
            listing = "\n".join(f"{i}. {self._step_label(s)}"
                                for i, s in enumerate(steps, 1))
            return out, (f"PIPELINE (manual override, {len(steps)} filters)\n"
                         f"{listing}\n"
                         f"Input: {'current result (chained)' if chain else 'original image'}\n")
        self._start_worker(task, f"Pipeline of {len(steps)} filters applied.")

    # -- file actions ----------------------------------------------------------

    def upload_image(self):
        if self.processing:
            messagebox.showinfo("Busy", "Please wait for the current job to finish.")
            return
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        try:
            image = read_image(path)
        except Exception as exc:
            messagebox.showerror("Image loading error", str(exc))
            return
        self.original = image
        self.restored = image.copy()
        self.reference = None
        self.input_path = path
        self.current_plan = []
        self.pipeline = []
        self._refresh_pipeline_list()
        self._refresh_previews()
        self._set_text(self.recommend_text, "Analysing the new image...")
        h, w = image.shape[:2]
        self._set_status(f"Loaded: {os.path.basename(path)} ({w} x {h}) - analysing...")
        self._show_analysis(f"Image: {os.path.basename(path)}\nDimensions: {w} x {h}\n"
                            f"Channels: {image.shape[2]}\n")
        # Analyse immediately; auto-restore unless the user turned it off.
        self.analyze_current(auto_apply=self.auto_var.get())

    def load_reference(self):
        path = filedialog.askopenfilename(
            title="Select ground-truth/reference image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        try:
            self.reference = read_image(path)
            self._set_status("Reference image loaded for full-reference metrics.")
            if self.restored is not None:
                self._show_metrics()
        except Exception as exc:
            messagebox.showerror("Reference loading error", str(exc))

    def _show_metrics(self):
        if self.reference is None or self.restored is None:
            return
        metrics = quality_metrics(self.reference, self.restored)
        self._set_text(self.analysis_text,
                       self._analysis_base.rstrip() +
                       "\n\nReference metrics:\n" + metrics)

    # -- background worker -----------------------------------------------------

    def _start_worker(self, task, success_message, on_success=None):
        if self.original is None:
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        if self.processing:
            return
        self.processing = True
        self.progress.start(12)
        self._set_status("Processing...")

        def worker():
            start = time.time()
            try:
                result, details = task()
                elapsed = time.time() - start
                self.root.after(0, lambda: self._finish_worker(
                    result, details, elapsed, success_message, on_success))
            except Exception as exc:
                self.root.after(0, lambda error=str(exc): self._worker_error(error))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_worker(self, result, details, elapsed, success_message, on_success):
        self.processing = False
        self.progress.stop()
        self.restored = result
        self._refresh_previews()
        self._show_analysis(details + f"\nProcessing time: {elapsed:.2f} seconds")
        if on_success is not None:
            on_success()
        if self.reference is not None:
            self._show_metrics()
        self._set_status(success_message)

    def _worker_error(self, error):
        self.processing = False
        self.progress.stop()
        self._set_status("Processing failed.")
        messagebox.showerror("Processing error", error)

    # -- automatic mode ----------------------------------------------------------

    def analyze_current(self, auto_apply=True):
        """Analyse the working image, recommend a plan and (optionally) apply it."""
        if self.original is None:
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        source = self.original.copy()
        holder = {}

        def task():
            info = analyze_image(source)
            plan = recommend_plan(info)
            holder["plan"] = plan
            holder["info"] = info
            details = format_analysis(info)
            if auto_apply:
                result = apply_plan(source, plan)
                if plan:
                    details += ("\nAPPLIED AUTOMATICALLY\n- " +
                                "\n- ".join(s.filter for s in plan) + "\n")
                else:
                    details += "\nNo degradation worth correcting; original kept.\n"
            else:
                result = source
                details += "\nAuto-restore is off. Press 'Automatic Restore' to apply.\n"
            details += "\nAutomatic detection is heuristic. Review the result.\n"
            return result, details

        def done():
            plan = holder.get("plan", [])
            self._show_recommendation(plan, holder.get("info"))
            self.pipeline = [Step(**vars(st)) for st in plan]
            self._refresh_pipeline_list(0 if plan else None)
            if plan:
                self._load_step_into_controls(plan[0])

        message = ("Analysis complete; recommended restoration applied."
                   if auto_apply else "Analysis complete; recommendation ready.")
        self._start_worker(task, message, done)

    def run_automatic(self):
        self.analyze_current(auto_apply=True)

    # -- manual override ---------------------------------------------------------

    def run_manual(self):
        if self.original is None:
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        name = self.filter_var.get()
        k = odd_kernel(self._safe_int(self.kernel_var, 5))
        strength = self._safe_float(self.strength_var, 1.0)
        psf = self.psf_var.get()
        angle = self._safe_int(self.angle_var, 0)
        chain = self.chain_var.get() and self.restored is not None
        source = (self.restored if chain else self.original).copy()

        def task():
            out = apply_filter(source, name, k, strength, psf, angle)
            kernel_txt = f"{k} x {k}" if name in KERNEL_FILTERS else "n/a"
            strength_txt = f"{strength:.2f}" if name in STRENGTH_FILTERS else "n/a"
            return out, (f"MANUAL PROCESSING (override)\nApplied filter: {name}\n"
                         f"Kernel size: {kernel_txt}\nStrength: {strength_txt}\n"
                         f"Input: {'current result (chained)' if chain else 'original image'}\n")
        self._start_worker(task, f"{name} applied.")

    def add_test_degradation(self, kind):
        if self.original is None:
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        if self.processing:
            return
        img = self.original.copy()
        if kind == "gaussian":
            noise = np.random.normal(0, 18, img.shape).astype(np.float32)
            img = np.uint8(np.clip(img.astype(np.float32) + noise, 0, 255))
            label = "Gaussian noise added"
        elif kind == "sp":
            amount = 0.035
            rnd = np.random.random(img.shape[:2])
            img[rnd < amount / 2] = 0
            img[rnd > 1 - amount / 2] = 255
            label = "Salt-and-pepper noise added"
        elif kind == "blur":
            img = cv2.GaussianBlur(img, (15, 15), 4)
            label = "Gaussian blur added"
        else:
            kernel = np.zeros((15, 15), np.float32)
            kernel[7, :] = 1.0 / 15
            img = cv2.filter2D(img, -1, kernel)
            label = "Horizontal motion blur added"
        self.original = img
        self.restored = img.copy()
        self.reference = None
        self._refresh_previews()
        self._show_analysis(label + "\nThe working image has been replaced. "
                            "Reset to return to the originally loaded image.\n")
        self._set_status(label + " (synthetic test image)")
        self.analyze_current(auto_apply=self.auto_var.get())

    def promote_result(self):
        if self.restored is None:
            return
        self.original = self.restored.copy()
        self._refresh_previews()
        self._set_status("Restored image is now the working input.")

    def save_result(self):
        if self.restored is None:
            messagebox.showwarning("No result", "Process an image first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save restored image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"),
                       ("TIFF", "*.tiff")]
        )
        if not path:
            return
        try:
            save_image(path, self.restored)
            self._set_status("Saved: " + os.path.basename(path))
            messagebox.showinfo("Saved", "Restored image saved successfully.")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def reset(self):
        if self.processing:
            return
        if self.input_path and os.path.exists(self.input_path):
            try:
                self.original = read_image(self.input_path)
                self.restored = self.original.copy()
                self.reference = None
                self.current_plan = []
                self.pipeline = []
                self._refresh_pipeline_list()
                self._refresh_previews()
                self._show_analysis("Workspace reset to the originally loaded image.")
                self._set_text(self.recommend_text,
                               "Press 'Automatic Restore' to analyse this image again.")
                self._set_status("Reset completed.")
            except Exception as exc:
                messagebox.showerror("Reset error", str(exc))
        else:
            self.original = None
            self.restored = None
            self.reference = None
            self.current_plan = []
            self.pipeline = []
            self._refresh_pipeline_list()
            self.original_label.configure(image="", text="No image loaded")
            self.restored_label.configure(image="", text="Restored result appears here")
            self._show_analysis("Upload an image to begin.")
            self._set_text(self.recommend_text, "")
            self._set_status("Workspace cleared.")


def main():
    root = tk.Tk()
    ImageRestorationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
