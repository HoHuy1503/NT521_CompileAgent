#!/bin/bash

# Build-Aware Compiler Wrapper for AutoCompiler / NT521_CompileAgent (Stage 3)
# Intercepts gcc / g++ / cc / c++ / clang / clang++ when ENABLE_COMPILER_WRAPPER is enabled
# (or WRAPPER_OPTI is non-empty) and routes compile-only, shared-library, and link flags safely.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 [source files and options]"
  exit 1
fi

real_gcc="gcc-real"
real_gxx="g++-real"
real_clang="clang-real"
real_clangxx="clang++-real"
real_cc="gcc-real"
real_cxx="g++-real"

compiler=""
case "$(basename "$0")" in
  gcc)
    compiler="${WRAPPER_GCC:-$real_gcc}"
    ;;
  clang)
    compiler="${WRAPPER_CLANG:-$real_clang}"
    ;;
  g++)
    compiler="${WRAPPER_GXX:-$real_gxx}"
    ;;
  clang++)
    compiler="${WRAPPER_CLANGXX:-$real_clangxx}"
    ;;
  cc)
    compiler="${WRAPPER_CC:-$real_cc}"
    ;;
  c++)
    compiler="${WRAPPER_CXX:-$real_cxx}"
    ;;
  *)
    echo "Error: Unknown compiler '$0'." >&2
    exit 1
    ;;
esac

# Pass-through when wrapper is disabled or unset with empty WRAPPER_OPTI
enable_lc="$(printf '%s' "$ENABLE_COMPILER_WRAPPER" | tr '[:upper:]' '[:lower:]')"
if [ "$enable_lc" = "0" ] || [ "$enable_lc" = "false" ] || [ "$enable_lc" = "no" ]; then
  exec "$compiler" "$@"
fi
if [ -z "$enable_lc" ] && [ -z "$WRAPPER_OPTI" ]; then
  exec "$compiler" "$@"
fi

# Inspect incoming arguments for mode detection
IS_QUERY_OR_PREPROC=false
IS_COMPILE_ONLY=false
IS_SHARED=false
IS_STATIC=false
IS_RELOCATABLE=false
HAS_FPIC=false
HAS_V_FLAG=false
HAS_INPUT_OR_OUT=false
HAS_NON_O0_IN_ARGS=false
HAS_ANY_OPTI_IN_ARGS=false

for arg in "$@"; do
  case "$arg" in
    -E|-M|-MM|-fsyntax-only|-dumpversion|-dumpmachine|-dumpspecs|--version|--help|-print-search-dirs|-print-libgcc-file-name|-print-multiarch|-print-file-name=*|-print-prog-name=*|-print-*)
      IS_QUERY_OR_PREPROC=true
      ;;
    -c|-S)
      IS_COMPILE_ONLY=true
      ;;
    -shared)
      IS_SHARED=true
      ;;
    -r|-Wl,-r|--relocatable)
      IS_RELOCATABLE=true
      ;;
    -static|-static-libgcc)
      if [ "$arg" = "-static" ]; then
        IS_STATIC=true
      fi
      ;;
    -fPIC|-fpic)
      HAS_FPIC=true
      ;;
    -v|-V)
      HAS_V_FLAG=true
      ;;
    -o)
      HAS_INPUT_OR_OUT=true
      ;;
    -O0)
      HAS_ANY_OPTI_IN_ARGS=true
      HAS_NON_O0_IN_ARGS=false
      ;;
    -O|-O1|-O2|-O3|-Os|-Ofast|-Og|-Oz)
      HAS_ANY_OPTI_IN_ARGS=true
      HAS_NON_O0_IN_ARGS=true
      ;;
    *.c|*.cc|*.cpp|*.cxx|*.C|*.S|*.s|*.i|*.ii|*.o|*.a|*.so|*.so.*)
      HAS_INPUT_OR_OUT=true
      ;;
  esac
done

# Query-only -v / -V invocation (e.g., configure probing compiler version)
if $HAS_V_FLAG && ! $HAS_INPUT_OR_OUT; then
  IS_QUERY_OR_PREPROC=true
fi

if $IS_QUERY_OR_PREPROC; then
  exec "$compiler" "$@"
fi

IS_LINK=true
if $IS_COMPILE_ONLY; then
  IS_LINK=false
fi

# Parse WRAPPER_OPTI as whitespace-separated tokens (preserve commas inside -Wl,... and -fsanitize=...)
read -ra OPTI_ARR <<< "$WRAPPER_OPTI"

HAS_WRAP_OPTI=false
WRAP_EFFECTIVE_O0=false
HAS_WRAP_NON_O0=false
HAS_WRAP_FORTIFY=false
HAS_WRAP_STACK_PROT=false
HAS_WRAP_PIE=false
HAS_WRAP_RELRO=false
HAS_WRAP_DEBUG=false
HAS_WRAP_SANITIZER=false

for opt in "${OPTI_ARR[@]}"; do
  case "$opt" in
    -O0)
      HAS_WRAP_OPTI=true
      WRAP_EFFECTIVE_O0=true
      HAS_WRAP_NON_O0=false
      ;;
    -O|-O1|-O2|-O3|-Os|-Ofast|-Og|-Oz)
      HAS_WRAP_OPTI=true
      WRAP_EFFECTIVE_O0=false
      HAS_WRAP_NON_O0=true
      ;;
    -D_FORTIFY_SOURCE*|-U_FORTIFY_SOURCE*)
      HAS_WRAP_FORTIFY=true
      ;;
    -fstack-protector*)
      HAS_WRAP_STACK_PROT=true
      ;;
    -pie|-fPIE|-fpie)
      HAS_WRAP_PIE=true
      ;;
    -Wl,-z,relro*|-Wl,-z,now*)
      HAS_WRAP_RELRO=true
      ;;
    -g|-g[0-9]|-gdwarf*|-ggdb*)
      HAS_WRAP_DEBUG=true
      ;;
    -fsanitize=*)
      HAS_WRAP_SANITIZER=true
      ;;
  esac
done

# Determine effective optimization status (FOR _FORTIFY_SOURCE safety)
EFFECTIVE_NON_O0=false
if $HAS_WRAP_OPTI; then
  if $HAS_WRAP_NON_O0; then
    EFFECTIVE_NON_O0=true
  fi
else
  if $HAS_NON_O0_IN_ARGS; then
    EFFECTIVE_NON_O0=true
  fi
fi

# Filter conflicting flags from original arguments
params=()
for arg in "$@"; do
  case "$arg" in
    -s)
      continue
      ;;
    -Werror|-Werror=*)
      continue
      ;;
    --debug|--debug=*)
      continue
      ;;
    -g|-gdwarf|-gdwarf-*)
      if $HAS_WRAP_DEBUG; then
        continue
      fi
      ;;
    -O|-O0|-O1|-O2|-O3|-Os|-Ofast|-Og|-Oz|--optimize|--optimize=*)
      if $HAS_WRAP_OPTI; then
        continue
      fi
      ;;
    -D_FORTIFY_SOURCE*|-U_FORTIFY_SOURCE*)
      if $HAS_WRAP_FORTIFY; then
        continue
      fi
      ;;
    -fno-stack-protector|-fstack-protector*)
      if $HAS_WRAP_STACK_PROT; then
        continue
      fi
      ;;
    -no-pie|-nopie|-fno-pie|-fno-PIE)
      if $HAS_WRAP_PIE; then
        continue
      fi
      ;;
    -Wl,-z,norelro|-Wl,-z,lazy)
      if $HAS_WRAP_RELRO; then
        continue
      fi
      ;;
    -Wl,--no-undefined|-Wl,-z,defs)
      if $IS_SHARED && $HAS_WRAP_SANITIZER; then
        continue
      fi
      ;;
    -Wl,*)
      if $IS_SHARED && $HAS_WRAP_SANITIZER; then
        cleaned_wl="${arg//,--no-undefined/}"
        cleaned_wl="${cleaned_wl//-Wl,--no-undefined,/-Wl,}"
        cleaned_wl="${cleaned_wl//,-z,defs/}"
        cleaned_wl="${cleaned_wl//-Wl,-z,defs,/-Wl,}"
        if [ "$cleaned_wl" = "-Wl," ] || [ "$cleaned_wl" = "-Wl" ] || [ -z "$cleaned_wl" ]; then
          continue
        fi
        params+=("$cleaned_wl")
        continue
      fi
      ;;
  esac
  params+=("$arg")
done

# Append wrapper options according to compile/link/shared/static context
added_undef_fortify=false
for opt in "${OPTI_ARR[@]}"; do
  [ -z "$opt" ] && continue
  case "$opt" in
    -pie)
      if $IS_LINK && ! $IS_SHARED && ! $IS_STATIC && ! $IS_RELOCATABLE; then
        params+=("$opt")
      fi
      ;;
    -fPIE|-fpie)
      if $IS_STATIC || $IS_RELOCATABLE; then
        continue
      elif $IS_COMPILE_ONLY || $IS_SHARED || $HAS_FPIC; then
        params+=("-fPIC")
      else
        params+=("$opt")
      fi
      ;;
    -Wl,*)
      if $IS_LINK && ! $IS_RELOCATABLE; then
        params+=("$opt")
      fi
      ;;
    -D_FORTIFY_SOURCE*)
      if $EFFECTIVE_NON_O0; then
        if ! $added_undef_fortify; then
          params+=("-U_FORTIFY_SOURCE")
          added_undef_fortify=true
        fi
        params+=("$opt")
      fi
      ;;
    -U_FORTIFY_SOURCE*)
      params+=("$opt")
      added_undef_fortify=true
      ;;
    *)
      params+=("$opt")
      ;;
  esac
done

exec "$compiler" "${params[@]}"
