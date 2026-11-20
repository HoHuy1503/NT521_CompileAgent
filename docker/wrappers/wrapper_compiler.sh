#!/bin/bash

# Compiler Wrapper (Ch3nYe/AutoCompiler)
# Usage:
# 1. Link this script to PATH, e.g., ln -s /wrapper_compiler.sh /usr/local/bin/gcc
# 2. Move original compiler to <compiler>-real (e.g., mv /usr/local/bin/gcc /usr/local/bin/gcc-real)
# 3. Set ENABLE_COMPILER_WRAPPER=True or WRAPPER_OPTI to enable option replacement
# Example: WRAPPER_OPTI="-O2 -gdwarf-4" gcc -O3 -Wall -o test.elf test.c

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 [source files and options]"
  echo "This is a wrapper for the compiler 'gcc, g++, clang, clang++, cc, c++'."
  echo "It modifies the compilation options and ignores certain flags."
  exit 1
fi

real_gcc="gcc-real"
real_gxx="g++-real"
real_clang="clang-real"
real_clangxx="clang++-real"
real_cc="gcc-real"
real_cxx="g++-real"

resolve_compiler() {
  case "$(basename "$1")" in
    gcc)
      echo "${WRAPPER_GCC:-$real_gcc}"
      ;;
    clang)
      echo "${WRAPPER_CLANG:-$real_clang}"
      ;;
    g++)
      echo "${WRAPPER_GXX:-$real_gxx}"
      ;;
    clang++)
      echo "${WRAPPER_CLANGXX:-$real_clangxx}"
      ;;
    cc)
      echo "${WRAPPER_CC:-$real_cc}"
      ;;
    c++)
      echo "${WRAPPER_CXX:-$real_cxx}"
      ;;
    *)
      echo "Error: Unknown compiler '$1'." >&2
      return 1
      ;;
  esac
}

compiler="$(resolve_compiler "$0")" || exit 1

# Pass-through directly if neither ENABLE_COMPILER_WRAPPER nor WRAPPER_OPTI is enabled
enable_lc="$(printf '%s' "$ENABLE_COMPILER_WRAPPER" | tr '[:upper:]' '[:lower:]')"
if [ "$enable_lc" = "0" ] || [ "$enable_lc" = "false" ] || [ "$enable_lc" = "no" ]; then
  exec "$compiler" "$@"
fi
if [ -z "$ENABLE_COMPILER_WRAPPER" ] && [ -z "$WRAPPER_OPTI" ]; then
  exec "$compiler" "$@"
fi

skip_base_arg() {
  local arg="$1"
  case "$arg" in
    -s|-DNDEBUG|-g|-gdwarf-*|-gdwarf|--debug|--debug=*|--optimize|--optimize=*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

skip_opti_arg() {
  local arg="$1"
  case "$arg" in
    -O|-O0|-O1|-O2|-O3|-Os|-Ofast|-Og|-Oz)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

IS_WRAP_OPTI=false
if [[ "$WRAPPER_OPTI" =~ -O[0-3sfastgz]* ]]; then
  IS_WRAP_OPTI=true
fi

params=()
for arg in "$@"; do
  if skip_base_arg "$arg"; then
    continue
  fi
  if $IS_WRAP_OPTI && skip_opti_arg "$arg"; then
    continue
  fi
  params+=("$arg")
done

if [ -n "$WRAPPER_OPTI" ]; then
  WRAPPER_OPTI_MODIFIED="${WRAPPER_OPTI//,/ }"
  read -ra OPTI_ARR <<< "$WRAPPER_OPTI_MODIFIED"
  for opt in "${OPTI_ARR[@]}"; do
    opt_trimmed="$(echo "$opt" | xargs)"
    if [ -n "$opt_trimmed" ]; then
      params+=("$opt_trimmed")
    fi
  done
fi

exec "$compiler" "${params[@]}"
