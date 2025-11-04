import torch


def save_ckpt(path, model, optimizer, scaler, ema, step: int, extra: dict = None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict(),
        "step": step}
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def load_ckpt(path, model, optimizer=None, scaler=None, ema=None, map_location="cuda"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if ema is not None and "ema" in state:
        ema.load_state_dict(state["ema"])
    return state.get("step", 0), state.get("extra", {})