from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


def generate_notebook_cam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class_index: int,
) -> np.ndarray:
    """
    Keep the classifier-weight CAM used by the Kaggle notebook.

    This function does not change the model, checkpoint, prediction,
    confidence, probabilities, or preprocessing.
    """
    model.eval()
    dec1_outputs: list[torch.Tensor] = []

    def forward_hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        dec1_outputs.append(output.detach().clone())

    hook_handle = model.dec1.register_forward_hook(forward_hook)

    try:
        with torch.no_grad():
            logits = model(input_tensor)
    finally:
        hook_handle.remove()

    if not dec1_outputs:
        raise RuntimeError("Could not capture the dec1 feature map.")

    class_index = int(target_class_index)

    if class_index < 0 or class_index >= int(logits.shape[1]):
        raise ValueError(
            f"Invalid target class index {class_index}; "
            f"the model returned {int(logits.shape[1])} classes."
        )

    feature_map = dec1_outputs[0][0]

    first_linear_weights = model.classifier[1].weight.detach()
    final_linear_weights = model.classifier[4].weight.detach()

    combined_weights = (
        final_linear_weights[class_index]
        @ first_linear_weights
    )

    cam = torch.einsum(
        "c,chw->hw",
        combined_weights,
        feature_map,
    )

    cam = F.relu(cam)

    cam = torch.nan_to_num(
        cam,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    cam_min = cam.min()
    cam_max = cam.max()

    if float(cam_max - cam_min) <= 1e-8:
        cam = feature_map.abs().mean(dim=0)
        cam_min = cam.min()
        cam_max = cam.max()

    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min + 1e-8
    )

    return (
        cam.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def _denormalize_input_tensor(
    input_tensor: torch.Tensor,
) -> np.ndarray:
    """
    Convert the normalized 224x224 model tensor back to RGB.
    """
    image = (
        input_tensor[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    image = (
        image * IMAGENET_STD
        + IMAGENET_MEAN
    )

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return np.uint8(
        np.round(image * 255.0)
    )


def _make_overlay(
    original_rgb: np.ndarray,
    cam: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Keep the notebook CAM display unchanged.
    """
    height, width = original_rgb.shape[:2]

    cam_resized = cv2.resize(
        np.asarray(cam, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    cam_resized = np.nan_to_num(
        cam_resized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    cam_resized = np.clip(
        cam_resized,
        0.0,
        1.0,
    )

    cam_uint8 = np.uint8(
        cam_resized * 255.0
    )

    heatmap_bgr = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    overlay_rgb = cv2.addWeighted(
        original_rgb,
        0.5,
        heatmap_rgb,
        0.5,
        0,
    )

    return (
        cam_resized,
        heatmap_rgb,
        overlay_rgb,
    )


def _robust_normalize(
    image: np.ndarray,
    mask: np.ndarray,
    low_percentile: float = 5.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Normalize an image using only values inside the brain mask.
    """
    values = image[
        mask > 0
    ]

    if values.size == 0:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    low = float(
        np.percentile(
            values,
            low_percentile,
        )
    )

    high = float(
        np.percentile(
            values,
            high_percentile,
        )
    )

    if high <= low + 1e-8:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    normalized = (
        image.astype(np.float32) - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    normalized *= (
        mask > 0
    ).astype(np.float32)

    return normalized.astype(
        np.float32
    )


def _brain_mask(
    original_rgb: np.ndarray,
) -> np.ndarray:
    """
    Estimate the visible head foreground and remove the outer
    skull and background.

    This mask is only used for bounding-box extraction.
    """
    height, width = original_rgb.shape[:2]

    gray = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    gray_blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    _, foreground = cv2.threshold(
        gray_blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
        ),
    )

    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.ones(
            (height, width),
            dtype=np.uint8,
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.drawContours(
        mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    coverage = (
        float(np.count_nonzero(mask))
        / float(height * width)
    )

    if coverage < 0.10 or coverage > 0.95:
        return np.ones(
            (height, width),
            dtype=np.uint8,
        )

    erosion_size = max(
        3,
        int(
            round(
                min(height, width)
                * 0.018
            )
        ),
    )

    if erosion_size % 2 == 0:
        erosion_size += 1

    inner_mask = cv2.erode(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                erosion_size,
                erosion_size,
            ),
        ),
        iterations=1,
    )

    if (
        np.count_nonzero(inner_mask)
        < 0.60 * np.count_nonzero(mask)
    ):
        inner_mask = mask

    return (
        inner_mask > 0
    ).astype(np.uint8)


def _build_lesion_saliency(
    original_rgb: np.ndarray,
    cam_resized: np.ndarray,
    brain_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Build an MRI-guided lesion map.

    CAM is used only as a weak prior. It does not modify the
    displayed heatmap.
    """
    gray_uint8 = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_uint8 = clahe.apply(
        gray_uint8
    )

    enhanced = (
        enhanced_uint8.astype(np.float32)
        / 255.0
    )

    local_background = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        sigmaX=6.0,
        sigmaY=6.0,
    )

    bright_anomaly = np.maximum(
        enhanced - local_background,
        0.0,
    )

    dark_anomaly = np.maximum(
        local_background - enhanced,
        0.0,
    )

    local_anomaly = np.maximum(
        bright_anomaly,
        0.55 * dark_anomaly,
    )

    top_hat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (17, 17),
    )

    white_top_hat = cv2.morphologyEx(
        enhanced_uint8,
        cv2.MORPH_TOPHAT,
        top_hat_kernel,
    ).astype(np.float32) / 255.0

    gradient_x = cv2.Sobel(
        enhanced,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gradient_y = cv2.Sobel(
        enhanced,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = cv2.magnitude(
        gradient_x,
        gradient_y,
    )

    local_anomaly = _robust_normalize(
        local_anomaly,
        brain_mask,
        10.0,
        99.5,
    )

    white_top_hat = _robust_normalize(
        white_top_hat,
        brain_mask,
        10.0,
        99.5,
    )

    gradient = _robust_normalize(
        gradient,
        brain_mask,
        20.0,
        99.0,
    )

    cam_prior = np.clip(
        cam_resized,
        0.0,
        1.0,
    )

    saliency = (
        0.52 * local_anomaly
        + 0.28 * white_top_hat
        + 0.12 * gradient
        + 0.08 * cam_prior
    )

    saliency *= (
        0.65
        + 0.35 * cam_prior
    )

    saliency *= brain_mask.astype(
        np.float32
    )

    saliency = cv2.GaussianBlur(
        saliency,
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0,
    )

    saliency = _robust_normalize(
        saliency,
        brain_mask,
        5.0,
        99.7,
    )

    return (
        saliency,
        local_anomaly,
    )


def _component_score(
    *,
    component_mask: np.ndarray,
    saliency: np.ndarray,
    anomaly: np.ndarray,
    cam_resized: np.ndarray,
    brain_distance: np.ndarray,
    area: int,
    box_width: int,
    box_height: int,
    brain_area: int,
) -> float:
    """
    Score a lesion candidate.
    """
    values = saliency[
        component_mask
    ]

    anomaly_values = anomaly[
        component_mask
    ]

    cam_values = cam_resized[
        component_mask
    ]

    distance_values = brain_distance[
        component_mask
    ]

    if values.size == 0:
        return -1.0

    mean_saliency = float(
        values.mean()
    )

    peak_saliency = float(
        values.max()
    )

    mean_anomaly = float(
        anomaly_values.mean()
    )

    mean_cam = float(
        cam_values.mean()
    )

    compactness = (
        float(area)
        / float(
            max(
                1,
                box_width * box_height,
            )
        )
    )

    area_ratio = (
        float(area)
        / float(
            max(
                1,
                brain_area,
            )
        )
    )

    size_score = float(
        np.exp(
            -(
                np.log(
                    area_ratio + 1e-8
                )
                - np.log(0.025)
            ) ** 2
            / (
                2.0 * 1.15 ** 2
            )
        )
    )

    border_score = float(
        np.clip(
            distance_values.mean()
            / 18.0,
            0.0,
            1.0,
        )
    )

    return (
        0.34 * mean_saliency
        + 0.18 * peak_saliency
        + 0.23 * mean_anomaly
        + 0.07 * mean_cam
        + 0.08 * compactness
        + 0.07 * size_score
        + 0.03 * border_score
    )


def _find_best_component(
    saliency: np.ndarray,
    anomaly: np.ndarray,
    cam_resized: np.ndarray,
    brain_mask: np.ndarray,
) -> tuple[
    np.ndarray | None,
    tuple[int, int] | None,
]:
    """
    Search several high-saliency thresholds and select
    the best lesion candidate.
    """
    height, width = saliency.shape

    brain_area = int(
        np.count_nonzero(
            brain_mask
        )
    )

    if brain_area <= 0:
        return None, None

    brain_distance = cv2.distanceTransform(
        np.uint8(
            brain_mask * 255
        ),
        cv2.DIST_L2,
        5,
    )

    values = saliency[
        brain_mask > 0
    ]

    if values.size < 20:
        return None, None

    best_score = -1.0
    best_mask = None
    best_peak = None

    for percentile in (
        97.0,
        95.0,
        93.0,
        91.0,
    ):
        threshold = float(
            np.percentile(
                values,
                percentile,
            )
        )

        binary = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        binary[
            (saliency >= threshold)
            & (brain_mask > 0)
        ] = 255

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3),
            ),
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 7),
            ),
        )

        (
            component_count,
            labels,
            stats,
            _,
        ) = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        for label_index in range(
            1,
            component_count,
        ):
            area = int(
                stats[
                    label_index,
                    cv2.CC_STAT_AREA,
                ]
            )

            area_ratio = (
                float(area)
                / float(brain_area)
            )

            if area_ratio < 0.0010:
                continue

            if area_ratio > 0.18:
                continue

            x = int(
                stats[
                    label_index,
                    cv2.CC_STAT_LEFT,
                ]
            )

            y = int(
                stats[
                    label_index,
                    cv2.CC_STAT_TOP,
                ]
            )

            box_width = int(
                stats[
                    label_index,
                    cv2.CC_STAT_WIDTH,
                ]
            )

            box_height = int(
                stats[
                    label_index,
                    cv2.CC_STAT_HEIGHT,
                ]
            )

            if box_width < 3 or box_height < 3:
                continue

            component_mask = (
                labels == label_index
            )

            score = _component_score(
                component_mask=component_mask,
                saliency=saliency,
                anomaly=anomaly,
                cam_resized=cam_resized,
                brain_distance=brain_distance,
                area=area,
                box_width=box_width,
                box_height=box_height,
                brain_area=brain_area,
            )

            if score <= best_score:
                continue

            component_values = np.where(
                component_mask,
                saliency,
                -1.0,
            )

            peak_y, peak_x = np.unravel_index(
                int(
                    np.argmax(
                        component_values
                    )
                ),
                component_values.shape,
            )

            best_score = score
            best_mask = component_mask
            best_peak = (
                int(peak_x),
                int(peak_y),
            )

    return (
        best_mask,
        best_peak,
    )


def _refine_component(
    *,
    best_mask: np.ndarray,
    best_peak: tuple[int, int],
    saliency: np.ndarray,
    anomaly: np.ndarray,
    brain_mask: np.ndarray,
) -> np.ndarray:
    """
    Expand the selected core to cover the visible lesion.
    """
    height, width = saliency.shape

    ys, xs = np.where(
        best_mask
    )

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    margin = max(
        10,
        int(
            round(
                min(height, width)
                * 0.07
            )
        ),
    )

    roi_x1 = max(
        0,
        x1 - margin,
    )

    roi_y1 = max(
        0,
        y1 - margin,
    )

    roi_x2 = min(
        width,
        x2 + margin + 1,
    )

    roi_y2 = min(
        height,
        y2 + margin + 1,
    )

    roi_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    roi_mask[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ] = 1

    seed_values = saliency[
        best_mask
    ]

    seed_anomaly = anomaly[
        best_mask
    ]

    saliency_threshold = max(
        0.22,
        float(
            np.percentile(
                seed_values,
                20,
            )
        ) * 0.55,
    )

    anomaly_threshold = max(
        0.18,
        float(
            np.percentile(
                seed_anomaly,
                20,
            )
        ) * 0.50,
    )

    region = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    region[
        (
            (saliency >= saliency_threshold)
            | (anomaly >= anomaly_threshold)
        )
        & (brain_mask > 0)
        & (roi_mask > 0)
    ] = 255

    region = cv2.morphologyEx(
        region,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        ),
    )

    region = cv2.morphologyEx(
        region,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
    )

    (
        _component_count,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        region,
        connectivity=8,
    )

    peak_x, peak_y = best_peak

    peak_label = int(
        labels[
            peak_y,
            peak_x,
        ]
    )

    if peak_label == 0:
        return best_mask

    refined_mask = (
        labels == peak_label
    )

    refined_area = int(
        stats[
            peak_label,
            cv2.CC_STAT_AREA,
        ]
    )

    brain_area = max(
        1,
        int(
            np.count_nonzero(
                brain_mask
            )
        ),
    )

    if refined_area / brain_area > 0.20:
        return best_mask

    return refined_mask

def _tight_localization_box(original_rgb, cam_resized):
    import numpy as np
    import cv2

    h, w = original_rgb.shape[:2]

    # STEP 1: smooth CAM slightly (removes noise)
    cam = cv2.GaussianBlur(cam_resized, (5, 5), 0)

    # STEP 2: adaptive threshold (MORE STABLE THAN BEFORE)
    mean = np.mean(cam)
    std = np.std(cam)

    threshold = mean + 1.2 * std  # 🔥 sharper focus than before

    binary = (cam > threshold).astype(np.uint8)

    # STEP 3: clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # STEP 4: find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, "No CAM region found"

    # STEP 5: biggest region (important)
    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)

    # STEP 6: SMART TIGHT PADDING (FIXED)
    pad_x = int(bw * 0.20)   # reduced from 0.35 → tighter box
    pad_y = int(bh * 0.20)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w - 1, x + bw + pad_x)
    y2 = min(h - 1, y + bh + pad_y)

    # STEP 7: prevent full-brain box
    area_ratio = ((x2 - x1) * (y2 - y1)) / (h * w)

    if area_ratio > 0.60:
        # fallback smaller centered region
        cx, cy = w // 2, h // 2
        size = min(h, w) // 4
        x1, y1 = cx - size, cy - size
        x2, y2 = cx + size, cy + size

    return (x1, y1, x2, y2), "Stable CAM-based localization"

def create_visualizations(
    input_tensor: torch.Tensor,
    cam: np.ndarray,
    create_tumor_box: bool,
) -> dict[str, Any]:
    """
    Keep the notebook CAM display and replace only the box extraction.
    """
    original_rgb = _denormalize_input_tensor(
        input_tensor
    )

    (
        cam_resized,
        raw_heatmap_rgb,
        overlay_rgb,
    ) = _make_overlay(
        original_rgb,
        cam,
    )

    bounding_box = None

    localization_reason = (
        "Bounding box disabled for this prediction."
    )

    if create_tumor_box:
        (
            bounding_box,
            localization_reason,
        ) = _tight_localization_box(
            original_rgb,
            cam_resized,
        )

    bbox_rgb = original_rgb.copy()
    combined_rgb = overlay_rgb.copy()

    if bounding_box is not None:
        x1, y1, x2, y2 = (
            bounding_box
        )

        cv2.rectangle(
            bbox_rgb,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.rectangle(
            combined_rgb,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    return {
        "original_rgb": original_rgb,
        "cam_resized": cam_resized,
        "raw_heatmap_rgb": raw_heatmap_rgb,
        "heatmap_rgb": overlay_rgb,
        "overlay_rgb": overlay_rgb,
        "bbox_rgb": bbox_rgb,
        "combined_rgb": combined_rgb,
        "bounding_box": bounding_box,
        "localization_reliable": (
            bounding_box is not None
        ),
        "localization_reason": (
            localization_reason
        ),
    }