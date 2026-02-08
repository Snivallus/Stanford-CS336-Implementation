import math


def cosine_annealing_scheduler(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """
    Compute the learning rate alpha_t using warmup + cosine annealing.

    Args:
        - t: Current training step (non-negative integer).
        - alpha_max: Maximum learning rate.
        - alpha_min: Minimum learning rate.
        - T_w: Number of warm-up iterations.
        - T_c: Number of cosine annealing iterations (end step of cosine schedule).

    Returns:
        - alpha_t: Learning rate at step t.
    """
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")
    if T_w < 0:
        raise ValueError(f"T_w must be non-negative, got {T_w}")
    if T_c < T_w:
        raise ValueError(f"T_c must be >= T_w, got T_c={T_c}, T_w={T_w}")
    if alpha_max < alpha_min:
        raise ValueError(f"alpha_max must be >= alpha_min, got {alpha_max} < {alpha_min}")

    # Warm-up phase.
    if t < T_w:
        if T_w == 0:
            return alpha_max
        return alpha_max * t / T_w

    # Cosine annealing phase.
    if t <= T_c:
        if T_c == T_w:
            return alpha_min
        progress = (t - T_w) / (T_c - T_w)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return alpha_min + cosine * (alpha_max - alpha_min)

    # Post-annealing phase.
    return alpha_min