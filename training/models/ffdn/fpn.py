"""Expose mmsegmentation's FPN to keep FFDN code identical to upstream."""

try:
    from mmseg.models.necks.fpn import FPN  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError('mmsegmentation must be installed to use the FFDN neck.') from exc


__all__ = ['FPN']
