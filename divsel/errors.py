"""Exception types shared across divsel.

Kept in their own module so every other module can import them without risking
an import cycle.
"""

from __future__ import annotations


class DivselError(Exception):
    """Base class for every error raised deliberately by divsel."""


class DescriptorMismatchError(DivselError):
    """Raised when two descriptor sets are not comparable.

    Seed descriptors (from ``--selected_images`` or a cached ``.npy``) may only be
    compared against candidate descriptors when every parameter that defines the
    feature space matches.  The message carries a field-by-field diff; see
    :func:`divsel.config.format_descriptor_diff`.
    """


class SpeciesError(DivselError):
    """A structure contains an element the descriptor is not defined for."""


class BoxError(DivselError):
    """A cell could not be built for a structure."""


class VacuumTooSmallError(BoxError):
    """The requested vacuum is too small for the descriptor cutoff.

    Raised only for a structure that actually goes through the boxing path --
    ``pbc`` false, a missing cell axis, or ``--cell_mode box``.  A fully
    periodic structure is never boxed, so the vacuum is irrelevant to it and
    this is never raised on its account.

    Deliberately distinct from its parent: this is a *configuration* error
    affecting every structure that needs boxing, so it aborts the run, whereas
    a plain :class:`BoxError` is a per-frame geometry failure that is recorded
    and skipped.
    """


class ScanError(DivselError):
    """Input files could not be enumerated or indexed."""
