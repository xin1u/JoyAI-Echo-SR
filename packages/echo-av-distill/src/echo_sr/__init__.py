"""Echo SR: Audio-Video Super-Resolution Training Framework for LTX-2.3."""

import logging

logger = logging.getLogger("echo_sr")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
