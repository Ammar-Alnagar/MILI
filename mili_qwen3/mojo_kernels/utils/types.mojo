"""
Minimal type definitions for MILI GPU kernels.
"""

# Basic data type
struct DType:
    var value: Int32

# Kernel status
struct KernelStatus:
    var success: Bool
    var error_code: Int32
