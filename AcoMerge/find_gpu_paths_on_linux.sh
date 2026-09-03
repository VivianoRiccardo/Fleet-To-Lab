for path in /usr/include /usr/local/include /usr/include/CL /usr/local/cuda/include /opt/rocm/opencl/include /opt/intel/opencl/include; do
  if [ -d "$path" ]; then
    echo "✓ EXISTS: $path"
  else
    echo "✗ MISSING: $path"
  fi
done
