# Create the OpenCL stub library
cat > /tmp/opencl_stub.c << 'EOF'
// Minimal OpenCL stub functions for auditwheel
void clGetPlatformIDs() {}
void clGetDeviceIDs() {}
void clCreateContext() {}
void clCreateCommandQueue() {}
void clCreateBuffer() {}
void clCreateProgramWithSource() {}
void clBuildProgram() {}
void clCreateKernel() {}
void clSetKernelArg() {}
void clEnqueueNDRangeKernel() {}
void clEnqueueReadBuffer() {}
void clEnqueueWriteBuffer() {}
void clFinish() {}
void clReleaseMemObject() {}
void clReleaseKernel() {}
void clReleaseProgram() {}
void clReleaseCommandQueue() {}
void clReleaseContext() {}
void clGetKernelWorkGroupInfo() {}
void clGetDeviceInfo() {}
void clReleaseDevice() {}
void clGetProgramBuildInfo() {}
void clEnqueueFillBuffer() {}
void clReleaseEvent() {}
void clWaitForEvents() {}
EOF

# Compile into shared library
gcc -shared -fPIC /tmp/opencl_stub.c -o libOpenCL.so.1

# Install it
cp libOpenCL.so.1 /usr/lib64/
ldconfig

# Verify it's installed
ldconfig -p | grep OpenCL
echo "Library installed:"
ls -la /usr/lib64/libOpenCL*
