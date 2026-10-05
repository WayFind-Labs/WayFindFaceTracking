from enum import Enum


class BiometricModality(str, Enum):
    """
    Supported biometric modalities across the WayFind platform.
    """
    FACE = "FACE"
    FINGERPRINT = "FINGERPRINT"
    BODY_MARK = "BODY_MARK"
    IRIS = "IRIS"

    @classmethod
    def from_str(cls, value: str) -> "BiometricModality":
        """Parse string to BiometricModality safely case-insensitively."""
        normalized = value.strip().upper()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unsupported biometric modality: '{value}'. "
            f"Allowed modalities: {[m.value for m in cls]}"
        )
