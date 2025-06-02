/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "tensorConvert.h"

// gpuTensorMean
template <typename T, bool isBGR>
__global__ void gpuTensorMean(float2 scale, T *input, int iWidth, float *output,
                              int oWidth, int oHeight, float3 mean_value) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= oWidth || y >= oHeight)
    return;

  const int n = oWidth * oHeight;
  const int m = y * oWidth + x;

  const int dx = ((float)x * scale.x);
  const int dy = ((float)y * scale.y);

  const T px = input[dy * iWidth + dx];

  const float3 rgb =
      isBGR ? make_float3(px.z, px.y, px.x) : make_float3(px.x, px.y, px.z);

  output[n * 0 + m] = rgb.x - mean_value.x;
  output[n * 1 + m] = rgb.y - mean_value.y;
  output[n * 2 + m] = rgb.z - mean_value.z;
}

template <bool isBGR>
cudaError_t launchTensorMean(void *input, imageFormat format, size_t inputWidth,
                             size_t inputHeight, float *output,
                             size_t outputWidth, size_t outputHeight,
                             const float3 &mean_value, cudaStream_t stream) {
  if (!input || !output)
    return cudaErrorInvalidDevicePointer;

  if (inputWidth == 0 || outputWidth == 0 || inputHeight == 0 ||
      outputHeight == 0)
    return cudaErrorInvalidValue;

  const float2 scale = make_float2(float(inputWidth) / float(outputWidth),
                                   float(inputHeight) / float(outputHeight));

  // launch kernel
  const dim3 blockDim(8, 8);
  const dim3 gridDim(iDivUp(outputWidth, blockDim.x),
                     iDivUp(outputHeight, blockDim.y));

  if (format == IMAGE_RGB8)
    gpuTensorMean<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (uchar3 *)input, inputWidth, output, outputWidth, outputHeight,
        mean_value);
  else if (format == IMAGE_RGBA8)
    gpuTensorMean<uchar4, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (uchar4 *)input, inputWidth, output, outputWidth, outputHeight,
        mean_value);
  else if (format == IMAGE_RGB32F)
    gpuTensorMean<float3, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (float3 *)input, inputWidth, output, outputWidth, outputHeight,
        mean_value);
  else if (format == IMAGE_RGBA32F)
    gpuTensorMean<float4, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (float4 *)input, inputWidth, output, outputWidth, outputHeight,
        mean_value);
  else
    return cudaErrorInvalidValue;

  return CUDA(cudaGetLastError());
}

// cudaTensorMeanRGB
cudaError_t cudaTensorMeanRGB(void *input, imageFormat format,
                              size_t inputWidth, size_t inputHeight,
                              float *output, size_t outputWidth,
                              size_t outputHeight, const float3 &mean_value,
                              cudaStream_t stream) {
  return launchTensorMean<false>(input, format, inputWidth, inputHeight, output,
                                 outputWidth, outputHeight, mean_value, stream);
}

// cudaTensorMeanBGR
cudaError_t cudaTensorMeanBGR(void *input, imageFormat format,
                              size_t inputWidth, size_t inputHeight,
                              float *output, size_t outputWidth,
                              size_t outputHeight, const float3 &mean_value,
                              cudaStream_t stream) {
  return launchTensorMean<true>(input, format, inputWidth, inputHeight, output,
                                outputWidth, outputHeight, mean_value, stream);
}

// gpuTensorNorm
template <typename T, bool isBGR>
__global__ void gpuTensorNorm(float2 scale, T *input, int iWidth, float *output,
                              int oWidth, int oHeight, float multiplier,
                              float min_value) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= oWidth || y >= oHeight)
    return;

  const int n = oWidth * oHeight;
  const int m = y * oWidth + x;

  const int dx = ((float)x * scale.x);
  const int dy = ((float)y * scale.y);

  const T px = input[dy * iWidth + dx];

  const float3 rgb =
      isBGR ? make_float3(px.z, px.y, px.x) : make_float3(px.x, px.y, px.z);

  output[n * 0 + m] = rgb.x * multiplier + min_value;
  output[n * 1 + m] = rgb.y * multiplier + min_value;
  output[n * 2 + m] = rgb.z * multiplier + min_value;
}

template <bool isBGR>
cudaError_t launchTensorNorm(void *input, imageFormat format, size_t inputWidth,
                             size_t inputHeight, float *output,
                             size_t outputWidth, size_t outputHeight,
                             const float2 &range, cudaStream_t stream) {
  if (!input || !output)
    return cudaErrorInvalidDevicePointer;

  if (inputWidth == 0 || outputWidth == 0 || inputHeight == 0 ||
      outputHeight == 0)
    return cudaErrorInvalidValue;

  const float2 scale = make_float2(float(inputWidth) / float(outputWidth),
                                   float(inputHeight) / float(outputHeight));

  const float multiplier = (range.y - range.x) / 255.0f;

  // launch kernel
  const dim3 blockDim(8, 8);
  const dim3 gridDim(iDivUp(outputWidth, blockDim.x),
                     iDivUp(outputHeight, blockDim.y));

  if (format == IMAGE_RGB8)
    gpuTensorNorm<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (uchar3 *)input, inputWidth, output, outputWidth, outputHeight,
        multiplier, range.x);
  else if (format == IMAGE_RGBA8)
    gpuTensorNorm<uchar4, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (uchar4 *)input, inputWidth, output, outputWidth, outputHeight,
        multiplier, range.x);
  else if (format == IMAGE_RGB32F)
    gpuTensorNorm<float3, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (float3 *)input, inputWidth, output, outputWidth, outputHeight,
        multiplier, range.x);
  else if (format == IMAGE_RGBA32F)
    gpuTensorNorm<float4, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (float4 *)input, inputWidth, output, outputWidth, outputHeight,
        multiplier, range.x);
  else
    return cudaErrorInvalidValue;

  return CUDA(cudaGetLastError());
}

// cudaTensorNormRGB
cudaError_t cudaTensorNormRGB(void *input, imageFormat format,
                              size_t inputWidth, size_t inputHeight,
                              float *output, size_t outputWidth,
                              size_t outputHeight, const float2 &range,
                              cudaStream_t stream) {
  return launchTensorNorm<false>(input, format, inputWidth, inputHeight, output,
                                 outputWidth, outputHeight, range, stream);
}

// cudaTensorNormBGR
cudaError_t cudaTensorNormBGR(void *input, imageFormat format,
                              size_t inputWidth, size_t inputHeight,
                              float *output, size_t outputWidth,
                              size_t outputHeight, const float2 &range,
                              cudaStream_t stream) {
  return launchTensorNorm<true>(input, format, inputWidth, inputHeight, output,
                                outputWidth, outputHeight, range, stream);
}

// gpuTensorNormMean
template <typename T, bool isBGR>
__global__ void
gpuTensorNormMean(T *input, int iWidth, float *output, int oWidth, int oHeight,
                  int stride, float2 scale, float multiplier, float min_value,
                  const float3 mean, const float3 stdDev) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= oWidth || y >= oHeight)
    return;

  const int m = y * oWidth + x;
  const int dx = ((float)x * scale.x);
  const int dy = ((float)y * scale.y);

  const T px = input[dy * iWidth + dx];

  const float3 rgb =
      isBGR ? make_float3(px.z, px.y, px.x) : make_float3(px.x, px.y, px.z);

  output[stride * 0 + m] =
      ((rgb.x * multiplier + min_value) - mean.x) / stdDev.x;
  output[stride * 1 + m] =
      ((rgb.y * multiplier + min_value) - mean.y) / stdDev.y;
  output[stride * 2 + m] =
      ((rgb.z * multiplier + min_value) - mean.z) / stdDev.z;
}

template <bool isBGR>
cudaError_t launchTensorNormMean(void *input, imageFormat format,
                                 size_t inputWidth, size_t inputHeight,
                                 float *output, size_t outputWidth,
                                 size_t outputHeight, const float2 &range,
                                 const float3 &mean, const float3 &stdDev,
                                 cudaStream_t stream, size_t channelStride) {
  if (!input || !output)
    return cudaErrorInvalidDevicePointer;

  if (inputWidth == 0 || outputWidth == 0 || inputHeight == 0 ||
      outputHeight == 0)
    return cudaErrorInvalidValue;

  if (channelStride == 0)
    channelStride = outputWidth * outputHeight;

  const float2 scale = make_float2(float(inputWidth) / float(outputWidth),
                                   float(inputHeight) / float(outputHeight));

  const float multiplier = (range.y - range.x) / 255.0f;

  // launch kernel
  const dim3 blockDim(8, 8);
  const dim3 gridDim(iDivUp(outputWidth, blockDim.x),
                     iDivUp(outputHeight, blockDim.y));

  if (format == IMAGE_RGB8)
    gpuTensorNormMean<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>(
        (uchar3 *)input, inputWidth, output, outputWidth, outputHeight,
        channelStride, scale, multiplier, range.x, mean, stdDev);
  else if (format == IMAGE_RGBA8)
    gpuTensorNormMean<uchar4, isBGR><<<gridDim, blockDim, 0, stream>>>(
        (uchar4 *)input, inputWidth, output, outputWidth, outputHeight,
        channelStride, scale, multiplier, range.x, mean, stdDev);
  else if (format == IMAGE_RGB32F)
    gpuTensorNormMean<float3, isBGR><<<gridDim, blockDim, 0, stream>>>(
        (float3 *)input, inputWidth, output, outputWidth, outputHeight,
        channelStride, scale, multiplier, range.x, mean, stdDev);
  else if (format == IMAGE_RGBA32F)
    gpuTensorNormMean<float4, isBGR><<<gridDim, blockDim, 0, stream>>>(
        (float4 *)input, inputWidth, output, outputWidth, outputHeight,
        channelStride, scale, multiplier, range.x, mean, stdDev);
  else
    return cudaErrorInvalidValue;

  return CUDA(cudaGetLastError());
}

// cudaTensorNormMeanRGB
cudaError_t cudaTensorNormMeanRGB(void *input, imageFormat format,
                                  size_t inputWidth, size_t inputHeight,
                                  float *output, size_t outputWidth,
                                  size_t outputHeight, const float2 &range,
                                  const float3 &mean, const float3 &stdDev,
                                  cudaStream_t stream, size_t channelStride) {
  return launchTensorNormMean<false>(input, format, inputWidth, inputHeight,
                                     output, outputWidth, outputHeight, range,
                                     mean, stdDev, stream, channelStride);
}

// cudaTensorNormMeanBGR
cudaError_t cudaTensorNormMeanBGR(void *input, imageFormat format,
                                  size_t inputWidth, size_t inputHeight,
                                  float *output, size_t outputWidth,
                                  size_t outputHeight, const float2 &range,
                                  const float3 &mean, const float3 &stdDev,
                                  cudaStream_t stream, size_t channelStride) {
  return launchTensorNormMean<true>(input, format, inputWidth, inputHeight,
                                    output, outputWidth, outputHeight, range,
                                    mean, stdDev, stream, channelStride);
}

// gpuTensorNormPadding
template <typename T, bool isBGR>
__global__ void
gpuTensorNormPadding(float2 scale, T *input, int iWidth, int iHeight,
                     float *output, int oWidth, int oHeight, float multiplier,
                     float min_value, int paddingLeft, int paddingTop,
                     int paddingRight, int paddingBottom, float3 paddingColor) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= oWidth || y >= oHeight)
    return;

  const int n = oWidth * oHeight;
  const int m = y * oWidth + x;

  // 패딩 영역인지 확인
  bool isPadding = false;
  float3 rgb;

  if (x < paddingLeft || x >= (oWidth - paddingRight) || y < paddingTop ||
      y >= (oHeight - paddingBottom)) {
    // 패딩 영역: 지정된 패딩 컬러 사용
    isPadding = true;
    rgb = paddingColor;
  } else {
    // 이미지 영역: 입력 이미지에서 픽셀 가져오기
    const int imageX = x - paddingLeft;
    const int imageY = y - paddingTop;

    // 입력 이미지를 이미지 영역에 맞게 스케일링
    const int dx = ((float)imageX * scale.x);
    const int dy = ((float)imageY * scale.y);

    // 경계 체크
    if (dx >= 0 && dx < iWidth && dy >= 0 && dy < iHeight) {
      const T px = input[dy * iWidth + dx];
      rgb =
          isBGR ? make_float3(px.z, px.y, px.x) : make_float3(px.x, px.y, px.z);
    } else {
      // 경계 밖이면 패딩 컬러 사용
      isPadding = true;
      rgb = paddingColor;
    }
  }

  // 정규화 적용
  if (isPadding) {
    // 패딩 영역은 이미 정규화된 값으로 간주 (패딩 컬러가 이미 range 내의 값)
    output[n * 0 + m] = rgb.x;
    output[n * 1 + m] = rgb.y;
    output[n * 2 + m] = rgb.z;
  } else {
    // 이미지 영역은 정규화 적용
    output[n * 0 + m] = rgb.x * multiplier + min_value;
    output[n * 1 + m] = rgb.y * multiplier + min_value;
    output[n * 2 + m] = rgb.z * multiplier + min_value;
  }
}

template <bool isBGR>
cudaError_t launchTensorNormPadding(
    void *input, imageFormat format, size_t inputWidth, size_t inputHeight,
    float *output, size_t outputWidth, size_t outputHeight, const float2 &range,
    size_t paddingLeft, size_t paddingTop, size_t paddingRight,
    size_t paddingBottom, const float3 &paddingColor, cudaStream_t stream) {
  if (!input || !output)
    return cudaErrorInvalidDevicePointer;

  if (inputWidth == 0 || outputWidth == 0 || inputHeight == 0 ||
      outputHeight == 0)
    return cudaErrorInvalidValue;

  // 패딩 크기 검증
  if (paddingLeft + paddingRight >= outputWidth ||
      paddingTop + paddingBottom >= outputHeight)
    return cudaErrorInvalidValue;

  // 이미지 영역 크기 계산
  const size_t imageWidth = outputWidth - paddingLeft - paddingRight;
  const size_t imageHeight = outputHeight - paddingTop - paddingBottom;

  // 스케일 계산 (입력 이미지를 이미지 영역에 맞춤)
  const float2 scale = make_float2(float(inputWidth) / float(imageWidth),
                                   float(inputHeight) / float(imageHeight));

  const float multiplier = (range.y - range.x) / 255.0f;

  // launch kernel
  const dim3 blockDim(8, 8);
  const dim3 gridDim(iDivUp(outputWidth, blockDim.x),
                     iDivUp(outputHeight, blockDim.y));

  if (format == IMAGE_RGB8)
    gpuTensorNormPadding<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (uchar3 *)input, inputWidth, inputHeight, output, outputWidth,
        outputHeight, multiplier, range.x, paddingLeft, paddingTop,
        paddingRight, paddingBottom, paddingColor);
  else if (format == IMAGE_RGBA8)
    gpuTensorNormPadding<uchar4, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (uchar4 *)input, inputWidth, inputHeight, output, outputWidth,
        outputHeight, multiplier, range.x, paddingLeft, paddingTop,
        paddingRight, paddingBottom, paddingColor);
  else if (format == IMAGE_RGB32F)
    gpuTensorNormPadding<float3, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (float3 *)input, inputWidth, inputHeight, output, outputWidth,
        outputHeight, multiplier, range.x, paddingLeft, paddingTop,
        paddingRight, paddingBottom, paddingColor);
  else if (format == IMAGE_RGBA32F)
    gpuTensorNormPadding<float4, isBGR><<<gridDim, blockDim, 0, stream>>>(
        scale, (float4 *)input, inputWidth, inputHeight, output, outputWidth,
        outputHeight, multiplier, range.x, paddingLeft, paddingTop,
        paddingRight, paddingBottom, paddingColor);
  else
    return cudaErrorInvalidValue;

  return CUDA(cudaGetLastError());
}

// cudaTensorNormPaddingRGB
cudaError_t cudaTensorNormPaddingRGB(
    void *input, imageFormat format, size_t inputWidth, size_t inputHeight,
    float *output, size_t outputWidth, size_t outputHeight, const float2 &range,
    size_t paddingLeft, size_t paddingTop, size_t paddingRight,
    size_t paddingBottom, const float3 &paddingColor, cudaStream_t stream) {
  return launchTensorNormPadding<false>(
      input, format, inputWidth, inputHeight, output, outputWidth, outputHeight,
      range, paddingLeft, paddingTop, paddingRight, paddingBottom, paddingColor,
      stream);
}

// cudaTensorNormPaddingBGR
cudaError_t cudaTensorNormPaddingBGR(
    void *input, imageFormat format, size_t inputWidth, size_t inputHeight,
    float *output, size_t outputWidth, size_t outputHeight, const float2 &range,
    size_t paddingLeft, size_t paddingTop, size_t paddingRight,
    size_t paddingBottom, const float3 &paddingColor, cudaStream_t stream) {
  return launchTensorNormPadding<true>(input, format, inputWidth, inputHeight,
                                       output, outputWidth, outputHeight, range,
                                       paddingLeft, paddingTop, paddingRight,
                                       paddingBottom, paddingColor, stream);
}
