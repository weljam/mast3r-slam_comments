#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <cuda/std/limits>

#define BLOCK 16

//强制内联函数,使用GPu设备,返回bool类型,判断像素坐标是否在图像范围内
__forceinline__ __device__ bool inside_image(int u, int v, int W, int H) {
  return v >= 0 && v < H && u >= 0 && u < W;
}

//强制内联函数,使用GPu设备,返回bool类型,将像素限制在图像范围内
__forceinline__ __device__ void clamp(float& x, const float min, const float max) {
  x = fmin(fmax(x, min), max);
}

// 模板函数，用于在CUDA核函数中处理不同的数据类型
template <typename scalar_t>
__global__ void refine_matches_kernel(
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> D11, // 第一张图像的特征张量
  const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> D21, // 第二张图像的特征张量
  const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> p1, // 第一张图像中的初始像素坐标
  torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> p1_new, // 第一张图像中的新像素坐标
  const int radius, // 搜索半径
  const int dilation_max // 最大膨胀系数
  )
{
  // 获取线程索引，利用批次索引和线程索引计算全局索引
  const uint64_t n = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t b = blockIdx.y;

  const int h = D11.size(1); // 图像高度
  const int w = D11.size(2); // 图像宽度
  const int fdim = D11.size(3); // 特征维度

  // 获取像素及其特征
  long u0 = p1[b][n][0];
  long v0 = p1[b][n][1];

  scalar_t max_score = ::cuda::std::numeric_limits<scalar_t>::min(); // 初始化最大得分
  long u_new = u0; // 初始化新像素坐标
  long v_new = v0; // 初始化新像素坐标

  // 逐步减小膨胀系数进行搜索
  for (int d=dilation_max; d>0; d--) {
    const int rd = radius*d; // 计算当前膨胀半径
    const int diam = 2*rd + 1; // 计算当前搜索直径
    for (int i=0; i<diam; i+=d) {
      for (int j=0; j<diam; j+=d) {
        const long u = u0 - rd + i; // 计算当前搜索像素的横坐标
        const long v = v0 - rd + j; // 计算当前搜索像素的纵坐标

        if (inside_image(u, v, w, h)) { // 判断像素是否在图像范围内
          scalar_t score = 0.0; // 初始化得分
          for (int k=0; k<fdim; k++) {
            score += D21[b][n][k] * D11[b][v][u][k]; // 计算特征匹配得分
          }

          if (score > max_score) { // 更新最大得分和对应的像素坐标
            max_score = score;
            u_new = u;
            v_new = v;
          }
    
        }
      }
    }
    // 更新搜索中心
    u0 = u_new;
    v0 = v_new;
  }

  // 保存新像素坐标
  p1_new[b][n][0] = u_new;
  p1_new[b][n][1] = v_new;
}


std::vector<torch::Tensor> refine_matches_cuda(
    torch::Tensor D11, // 第一张图像的特征张量
    torch::Tensor D21, // 第二张图像的特征张量
    torch::Tensor p1, // 第一张图像中的初始像素坐标
    const int radius, // 搜索半径
    const int dilation) // 膨胀系数
  {
    const auto batch_size = p1.size(0); // 批量大小
    const auto n = p1.size(1); // 每批次的像素数量

    // 定义块和线程的数量
    const dim3 blocks((n + BLOCK - 1) / BLOCK, 
            batch_size);
    const dim3 threads(BLOCK);

    auto opts = p1.options(); // 获取p1张量的选项（如数据类型、设备等）
    torch::Tensor p1_new = torch::zeros(
    {batch_size, n, 2}, opts); // 创建新的像素坐标张量

    // 调用CUDA核函数，使用AT_DISPATCH_FLOATING_TYPES_AND_HALF宏来处理不同的数据类型
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(D11.type(), "refine_matches_kernel", ([&] {
    refine_matches_kernel<scalar_t><<<blocks, threads>>>(
      D11.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(), // 将D11张量转换为4维张量的访问器
      D21.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), // 将D21张量转换为3维张量的访问器
      p1.packed_accessor32<long,3,torch::RestrictPtrTraits>(), // 将p1张量转换为3维张量的访问器
      p1_new.packed_accessor32<long,3,torch::RestrictPtrTraits>(), // 将p1_new张量转换为3维张量的访问器
      radius, // 搜索半径
      dilation // 膨胀系数
    );
    }));

    return {p1_new}; // 返回新的像素坐标

}


__global__ void iter_proj_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> rays_img,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> pts_3d_norm,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> p_init,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> p_new,
    torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> converged,
    const int max_iter,
    const float lambda_init,
    const float cost_thresh
    )
{
  // 获得线程索引 ,利用批次索引和线程索引计算全局索引
  // blockIdx.x：这是当前线程所在的块的索引。CUDA 中的线程是按块（block）组织的，每个块中包含多个线程。
  // blockDim.x：这是每个块中线程的数量。它表示一个块中有多少个线程。
  // threadIdx.x：这是当前线程在其所在块中的索引。它表示线程在块中的位置。
  const uint64_t n = blockIdx.x * blockDim.x + threadIdx.x;
  //获得批次索引
  const uint64_t b = blockIdx.y;

  const int h = rays_img.size(1);
  const int w = rays_img.size(2);
  const int c = rays_img.size(3); // 通道数

  // 获取像素坐标
  float u = p_init[b][n][0];
  float v = p_init[b][n][1];

  // 如果初始坐标超出范围，进行修正
  clamp(u, 1, w-2);
  clamp(v, 1, h-2);

  // 设置射线和梯度
  float r[3];
  float gx[3];
  float gy[3];
  float err[3];

  float lambda = lambda_init;
  for (int i=0; i<max_iter; i++) {
    // 双线性插值
    int u11 = static_cast<int>(floor(u)); //向下取整
    int v11 = static_cast<int>(floor(v)); //向下取整
    float du = u - static_cast<float>(u11); //计算差值
    float dv = v - static_cast<float>(v11); //计算差值

      // 修正后的坐标确保双线性插值计算是安全的
      float w11 = du * dv; // 左上
      float w12 = (1.0-du) * dv; // 右上
      float w21 = du * (1.0-dv); // 左下
      float w22 = (1.0-du) * (1.0-dv); // 右下

      // 注意：像素坐标与区域计算相反
      float const* r11 = &rays_img[b][v11+1][u11+1][0]; // 右下
      float const* r12 = &rays_img[b][v11+1][u11][0]; // 左下
      float const* r21 = &rays_img[b][v11][u11+1][0]; // 右上
      float const* r22 = &rays_img[b][v11][u11][0]; // 左上

    // 进行插值
    #pragma unroll
    for (int j=0; j<3; j++) {
      r[j] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j];
    }
    #pragma unroll
    for (int j=3; j<6; j++) {
      gx[j-3] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j];
    }
    #pragma unroll
    for (int j=6; j<9; j++) {
      gy[j-6] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j];
    }

    // 归一化射线
    float r_norm = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    float r_norm_inv = 1.0/r_norm;
    #pragma unroll
    for (int j=0; j<3; j++) {
      r[j] *= r_norm_inv;
    }

    // 计算点图射线误差
    #pragma unroll
    for (int j=0; j<3; j++) {
      err[j] = r[j] - pts_3d_norm[b][n][j];
    }
    float cost = err[0]*err[0] + err[1]*err[1] + err[2]*err[2];

    // 设置系统
    // J^T J
    float A00 = gx[0]*gx[0] + gx[1]*gx[1] + gx[2]*gx[2];
    float A01 = gx[0]*gy[0] + gx[1]*gy[1] + gx[2]*gy[2];
    float A11 = gy[0]*gy[0] + gy[1]*gy[1] + gy[2]*gy[2];
    // - J^T r
    float b0 = - (err[0]*gx[0] + err[1]*gx[1] + err[2]*gx[2]);
    float b1 = - (err[0]*gy[0] + err[1]*gy[1] + err[2]*gy[2]);
    // LM 对角线
    A00 += lambda;
    A11 += lambda;

    // 解系统
    float det_inv = 1.0/(A00*A11 - A01*A01);
    float delta_u = det_inv * ( A11*b0 - A01*b1);
    float delta_v = det_inv * (-A01*b0 + A00*b1);

    // 获取新的像素坐标
    float u_new = u + delta_u;
    float v_new = v + delta_v;
    clamp(u_new, 1, w-2);
    clamp(v_new, 1, h-2);

    // 进行新的像素坐标的双线性插值
    u11 = static_cast<int>(floor(u_new));
    v11 = static_cast<int>(floor(v_new));
    du = u_new - u11;
    dv = v_new - v11;

    w11 = du * dv; // 左上
    w12 = (1.0-du) * dv; // 右上
    w21 = du * (1.0-dv); // 左下
    w22 = (1.0-du) * (1.0-dv); // 右下

    // 注意：像素坐标与区域计算相反
    r11 = &rays_img[b][v11+1][u11+1][0]; // 右下
    r12 = &rays_img[b][v11+1][u11][0]; // 左下
    r21 = &rays_img[b][v11][u11+1][0]; // 右上
    r22 = &rays_img[b][v11][u11][0]; // 左上

    //进行插值
    #pragma unroll //并行化
    for (int j=0; j<3; j++) {
      r[j] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j];
    }
    r_norm = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    r_norm_inv = 1.0/r_norm;
    #pragma unroll
    for (int j=0; j<3; j++) {
      r[j] *= r_norm_inv;
    }
    // 计算误差
    #pragma unroll
    for (int j=0; j<3; j++) {
      err[j] = r[j] - pts_3d_norm[b][n][j];
    }
    float new_cost = err[0]*err[0] + err[1]*err[1] + err[2]*err[2];

    // 更新像素坐标和lambda
    if (new_cost < cost) {
      u = u_new;
      v = v_new;
      lambda *= 0.1;
      converged[b][n] = new_cost < cost_thresh;
    }
    else {
      lambda *= 10.0;
      converged[b][n] = cost < cost_thresh;
    }

  }

  p_new[b][n][0] = u;
  p_new[b][n][1] = v;

}



std::vector<torch::Tensor> iter_proj_cuda(
  torch::Tensor rays_img_with_grad, // 包含梯度的图像射线
  torch::Tensor pts_3d_norm, // 归一化的3D点
  torch::Tensor p_init, // 初始像素坐标
  const int max_iter, // 最大迭代次数
  const float lambda_init, // 初始lambda值
  const float cost_thresh) // 代价阈值
{
  const auto batch_size = p_init.size(0); // 批量大小
  const auto n = p_init.size(1); // 每批次的像素数量

  const dim3 blocks((n + BLOCK - 1) / BLOCK, 
          batch_size); // 块的数量
  
  const dim3 threads(BLOCK); // 每个块的线程数量

  auto opts = p_init.options(); // p_init张量的选项（如数据类型、设备等）
  torch::Tensor p_new = torch::zeros(
  {batch_size, n, 2}, opts); // 创建新的像素坐标张量

  auto opts_bool = opts.dtype(torch::kBool); // 将保存的选项类型更改为布尔类型
  torch::Tensor converged = torch::zeros(
  {batch_size, n}, opts_bool); // 创建收敛标志张量

  // 调用CUDA核函数,定义块和线程数量
  iter_proj_kernel<<<blocks, threads>>>(
  rays_img_with_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),//将张量转换为4维张量的浮点数访问器
  pts_3d_norm.packed_accessor32<float,3,torch::RestrictPtrTraits>(), //张量转换为一个 3 维的浮点型访问器。
  p_init.packed_accessor32<float,3,torch::RestrictPtrTraits>(), //张量转换为一个 3 维的浮点型访问器。
  p_new.packed_accessor32<float,3,torch::RestrictPtrTraits>(), //张量转换为一个 3 维的浮点型访问器。
  converged.packed_accessor32<bool,2,torch::RestrictPtrTraits>(), //张量转换为一个 2 维的布尔型访问器。
  max_iter,
  lambda_init,
  cost_thresh
  );

  return {p_new, converged}; // 返回新的像素坐标和收敛标志

}