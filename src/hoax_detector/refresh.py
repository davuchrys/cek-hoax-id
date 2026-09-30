"""Periodic refresh: pull new feed items, retrain, hot-swap the served model."""
import logging
import threading
from pathlib import Path
from typing import Callable

from .data_update import cmd_fetch
from .train import train

log = logging.getLogger("hoax_detector.refresh")


def refresh(corpus_path: Path, model_path: Path) -> dict:
    cmd_fetch(corpus_path)
    return train(corpus_path, model_path)


def start_background_refresh(
    hours: float, corpus_path: Path, model_path: Path, on_new_model: Callable[[], None]
) -> threading.Event:
    """Run refresh() every `hours` hours in a daemon thread. Set the returned event to stop."""
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(hours * 3600):
            try:
                metrics = refresh(corpus_path, model_path)
                on_new_model()
                log.info("refreshed model: %s train rows, F1=%.3f", metrics["n_train"], metrics["f1"])
            except Exception:
                log.exception("scheduled refresh failed; keeping the current model")

    threading.Thread(target=loop, name="model-refresh", daemon=True).start()
    return stop
