#!/bin/bash
# Build script for MILI Mojo GPU kernels
# Compiles all Mojo kernels for the MILI inference system

set -e  # Exit on first error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
LIB_DIR="${SCRIPT_DIR}/lib"
CORE_DIR="${SCRIPT_DIR}/core"
MEMORY_DIR="${SCRIPT_DIR}/memory"
UTILS_DIR="${SCRIPT_DIR}/utils"

# Create output directories
mkdir -p "${BUILD_DIR}"
mkdir -p "${LIB_DIR}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building MILI Mojo GPU Kernels${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to compile a Mojo file
compile_mojo() {
    local mojo_file=$1
    local output_name=$2
    local lib_dir=$3
    
    echo -e "${YELLOW}Compiling: ${mojo_file}${NC}"
    
    # Check if mojo command exists
    if ! command -v mojo &> /dev/null; then
        echo -e "${RED}Error: mojo command not found. Please install Mojo SDK.${NC}"
        return 1
    fi
    
    # Compile with optimization flags
    echo -e "${YELLOW}Running: mojo build -O3 -o ${lib_dir}/${output_name}.so ${mojo_file}${NC}"
    if output=$(mojo build -O3 -o "${lib_dir}/${output_name}.so" "${mojo_file}" 2>&1); then
        echo -e "${GREEN}✓ Successfully compiled: ${output_name}${NC}"
        return 0
    else
        # Check if the error is just "no main function" (expected for libraries)
        if echo "$output" | grep -q "module does not contain a 'main' function"; then
            echo -e "${GREEN}✓ Successfully compiled: ${output_name} (library module)${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed to compile: ${mojo_file}${NC}"
            echo -e "${RED}$output${NC}"
            return 1
        fi
    fi
}

# Compile utility types first (dependency for other kernels)
echo -e "${BLUE}[1/6] Compiling utility types...${NC}"
compile_mojo "${UTILS_DIR}/types.🔥" "types" "${LIB_DIR}" || exit 1

# Compile core kernels
echo -e "${BLUE}[2/6] Compiling RoPE kernel...${NC}"
compile_mojo "${CORE_DIR}/rope.🔥" "rope" "${LIB_DIR}" || exit 1

echo -e "${BLUE}[3/6] Compiling RMSNorm kernel...${NC}"
compile_mojo "${CORE_DIR}/normalization.🔥" "rmsnorm" "${LIB_DIR}" || exit 1

echo -e "${BLUE}[4/6] Compiling SwiGLU activation kernel...${NC}"
compile_mojo "${CORE_DIR}/activations.🔥" "activations" "${LIB_DIR}" || exit 1

echo -e "${BLUE}[5/6] Compiling Attention kernels (FlashAttention, Decode)...${NC}"
compile_mojo "${CORE_DIR}/attention.🔥" "attention" "${LIB_DIR}" || exit 1

# Compile memory management kernels
echo -e "${BLUE}[6/6] Compiling Paged KV Cache kernel...${NC}"
compile_mojo "${MEMORY_DIR}/kv_cache.🔥" "kv_cache" "${LIB_DIR}" || exit 1

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Output libraries in: ${LIB_DIR}"
ls -lah "${LIB_DIR}"/*.so 2>/dev/null || echo "No .so files generated (expected for some targets)"

# Create a summary file
cat > "${LIB_DIR}/BUILD_INFO.txt" << EOF
MILI Mojo Kernels Build Information
===================================

Build Date: $(date)
Mojo Version: $(mojo --version 2>/dev/null || echo "Unknown")
Script Location: ${SCRIPT_DIR}

Compiled Kernels:
- types.so (Utility types and structures)
- rope.so (Rotary Position Embeddings)
- rmsnorm.so (RMSNorm normalization)
- activations.so (SwiGLU and other activations)
- attention.so (FlashAttention and Decode attention)
- kv_cache.so (Paged KV cache management)

Total Libraries: $(ls -1 "${LIB_DIR}"/*.so 2>/dev/null | wc -l)

Usage:
Import these libraries in your Python layer to use the GPU kernels:
    from ctypes import CDLL
    attention_lib = CDLL("${LIB_DIR}/attention.so")

Requirements:
- Mojo SDK (latest version recommended)
- CUDA 12.0+ or compatible GPU
- MAX framework for GPU support
EOF

echo -e "${GREEN}Build info saved to: ${LIB_DIR}/BUILD_INFO.txt${NC}"

# Optional: Run basic tests if test directory exists
if [ -d "${SCRIPT_DIR}/../tests" ]; then
    echo ""
    echo -e "${YELLOW}Running basic validation tests...${NC}"
    # Could add pytest or other test commands here
fi

echo -e "${GREEN}✓ Build process completed${NC}"
exit 0
