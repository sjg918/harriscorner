
//original code:
//https://github.com/qa276390/Cuda-Harris-Corner-Detector

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <torch/extension.h>

void parallelHarrisCornerDetectorCudaLauncher(
    float* grayImageArray, float* RHost, int imageWidth, int imageHeight
);

at::Tensor hcd(at::Tensor gray, int rows, int cols) {
    float* gray_ = gray.contiguous().data_ptr<float>();
    auto Rhost = at::zeros({rows * cols}, torch::dtype(torch::kFloat32));
    float* RHost_ = Rhost.data_ptr<float>();
    int rows_ = rows;
    int cols_ = cols;

    parallelHarrisCornerDetectorCudaLauncher(
        gray_, RHost_, rows_, cols_
    );

    return Rhost;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hcd", &hcd,
        "harris corner detection on gpu");
}
