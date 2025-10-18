// This file is modified from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/csrc/cuda/SigmoidFocalLoss_cuda.cu
// Jiajun Tang
// yelantingfeng@sjtu.edu.cn
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cfloat>

// Replace THC macros with modern equivalents
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// Replace THCCeilDiv
inline int64_t CeilDiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// Replace THCudaCheck
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    AT_ASSERTM(error == cudaSuccess, "CUDA error: ", cudaGetErrorString(error)); \
  } while(0)


template <typename T>
__global__ void SigmoidFocalLossForward(const int nthreads,
    const T* logits,
    const T* targets,
    const int num_classes,
    const float gamma,
    const float alpha,
    T* losses) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {

    // Decide it is positive or negative case.
    T c1 = targets[i];
    T c2 = (1.0 - c1);

    // alpha flag.
    T af1 = (alpha >= 0);
    T af2 = (1.0 - af1);

    T zn = (1.0 - alpha) * af1 + af2;
    T zp = (alpha) * af1 + af2;

    // p = 1. / 1. + expf(-x); p = sigmoid(x)
    T  p = 1. / (1. + expf(-logits[i]));

    // (1-p)**gamma * log(p) where
    T term1 = powf((1. - p), gamma) * logf(max(p, FLT_MIN));

    // p**gamma * log(1-p)
    T term2 = powf(p, gamma) *
            (-1. * logits[i] * (logits[i] >= 0) -
             logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0))));

    losses[i] = 0.0;
    losses[i] += -c1 * term1 * zp;
    losses[i] += -c2 * term2 * zn;

  } // CUDA_1D_KERNEL_LOOP
} // SigmoidFocalLossForward


template <typename T>
__global__ void SigmoidFocalLossBackward(const int nthreads,
    const T* logits,
    const T* targets,
    const T* d_losses,
    const int num_classes,
    const float gamma,
    const float alpha,
    T* d_logits) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {

    // Decide it is positive or negative case.
    T c1 = targets[i];
    T c2 = (1.0 - c1);

    // alpha flag.
    T af1 = (alpha >= 0);
    T af2 = (1.0 - af1);

    T zn = (1.0 - alpha) * af1 + af2;
    T zp = (alpha) * af1 + af2;

    // p = 1. / 1. + expf(-x); p = sigmoid(x)
    T  p = 1. / (1. + expf(-logits[i]));

    // (1-p)**gamma
    T pow_g_onemp = powf((1. - p), gamma);

    // d[ (1-p)**gamma ] / dx
    T d_pow_g_onemp = -gamma * powf((1. - p), (gamma - 1.0)) * p * (1.0 - p);

    // (1-p)
    T term1 = (1. - p);

    // d[ (1-p) ] / dx
    T d_term1 = -p * (1.0 - p);

    // p**gamma
    T pow_g_p = powf(p, gamma);

    // d[ p**gamma ] / dx
    T d_pow_g_p = gamma * powf(p, gamma - 1.0) * p * (1.0 - p);

    // [log(p)]
    T term2 = logf(max(p, FLT_MIN));

    // d[log(p)] / dx
    T d_term2 = (1.0 - p);

    // log(1-p)
    T term3 = logf(max(term1, FLT_MIN));

    // d[log(1-p)] / dx
    T d_term3 = -p;

    d_logits[i] = 0.0;
    // gradient of cross entropy loss: FL(pt) = -log(pt)
    d_logits[i] += -c1 * (d_pow_g_onemp * term2 + pow_g_onemp * d_term2) * zp;
    d_logits[i] += -c2 * (d_pow_g_p * term3 + pow_g_p * d_term3) * zn;
    // normalization
    d_logits[i] = d_logits[i] * d_losses[i];

  } // CUDA_1D_KERNEL_LOOP
} // SigmoidFocalLossBackward


at::Tensor SigmoidFocalLoss_forward_cuda(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const int num_classes,
		const float gamma,
		const float alpha) {
  AT_ASSERTM(logits.is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(logits.numel() == targets.numel(),
      "logits and targets should have same number of elements.");

  at::cuda::CUDAGuard device_guard(logits.device());

  auto losses = at::zeros({logits.numel()}, logits.options());
  auto losses_size = logits.numel();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(CeilDiv(losses_size, 512L), 4096L));
  dim3 block(512);

  if (losses.numel() == 0) {
    CUDA_CHECK(cudaGetLastError());
    return losses;
  }

  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "SigmoidFocalLoss_forward", [&] {
    SigmoidFocalLossForward<scalar_t><<<grid, block, 0, stream>>>(
         losses_size,
         logits.contiguous().data_ptr<scalar_t>(),
         targets.contiguous().data_ptr<scalar_t>(),
         num_classes,
         gamma,
         alpha,
         losses.data_ptr<scalar_t>());
  });
  CUDA_CHECK(cudaGetLastError());
  return losses;
}


at::Tensor SigmoidFocalLoss_backward_cuda(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const at::Tensor& d_losses,
		const int num_classes,
		const float gamma,
		const float alpha) {
  AT_ASSERTM(logits.is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(d_losses.is_cuda(), "d_losses must be a CUDA tensor");

  AT_ASSERTM(logits.numel() == targets.numel(),
      "logits and targets should have same number of elements");
  AT_ASSERTM(logits.dim() == 2, "logits should be NxClass");

  at::cuda::CUDAGuard device_guard(logits.device());

  auto d_logits = at::zeros({logits.size(0), logits.size(1)}, logits.options());
  auto d_logits_size = logits.numel();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(CeilDiv(d_logits_size, 512L), 4096L));
  dim3 block(512);

  if (d_logits.numel() == 0) {
    CUDA_CHECK(cudaGetLastError());
    return d_logits;
  }

  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "SigmoidFocalLoss_backward", [&] {
    SigmoidFocalLossBackward<scalar_t><<<grid, block, 0, stream>>>(
         d_logits_size,
         logits.contiguous().data_ptr<scalar_t>(),
         targets.contiguous().data_ptr<scalar_t>(),
         d_losses.contiguous().data_ptr<scalar_t>(),
         num_classes,
         gamma,
         alpha,
         d_logits.data_ptr<scalar_t>());
  });
  CUDA_CHECK(cudaGetLastError());
  return d_logits;
}
