#!/usr/bin/env python3
"""Basic Python environment test script"""

import sys
import math

print(f"Python version: {sys.version}")
print("Basic Python working correctly!")

# Test basic Python functionality without external dependencies
test_list = [[i, j, k] for i in range(2) for j in range(2) for k in range(2)]
print(f"Generated 3D point list with {len(test_list)} points")
print(f"Sample points: {test_list[:3]}")

# Test basic math operations
result = sum([math.sqrt(x**2 + y**2 + z**2) for x, y, z in test_list])
print(f"Sum of distances from origin: {result:.2f}")
print("Environment test completed successfully!")

# Try to import numpy if available
try:
    import numpy as np
    print(f"NumPy is available: {np.__version__}")
except ImportError:
    print("NumPy not available - will need to install it")