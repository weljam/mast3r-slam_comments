#include "gn.h"

// 高斯牛顿点调整
std::vector<torch::Tensor> gauss_newton_points(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_point,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh) {

  CHECK_CONTIGUOUS(Twc);
  CHECK_CONTIGUOUS(Xs);
  CHECK_CONTIGUOUS(Cs);
  CHECK_CONTIGUOUS(ii);
  CHECK_CONTIGUOUS(jj);
  CHECK_CONTIGUOUS(idx_ii2jj);
  CHECK_CONTIGUOUS(valid_match);
  CHECK_CONTIGUOUS(Q);

  // 调用 CUDA 实现的高斯牛顿点调整
  return gauss_newton_points_cuda(Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
      sigma_point, C_thresh, Q_thresh, max_iter, delta_thresh);
}

// 高斯牛顿射线调整
std::vector<torch::Tensor> gauss_newton_rays(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_ray,
  const float sigma_dist,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh) {

  CHECK_CONTIGUOUS(Twc);
  CHECK_CONTIGUOUS(Xs);
  CHECK_CONTIGUOUS(Cs);
  CHECK_CONTIGUOUS(ii);
  CHECK_CONTIGUOUS(jj);
  CHECK_CONTIGUOUS(idx_ii2jj);
  CHECK_CONTIGUOUS(valid_match);
  CHECK_CONTIGUOUS(Q);

  // 调用 CUDA 实现的高斯牛顿射线调整
  return gauss_newton_rays_cuda(Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
      sigma_ray, sigma_dist, C_thresh, Q_thresh, max_iter, delta_thresh);
}

// 高斯牛顿校准调整
std::vector<torch::Tensor> gauss_newton_calib(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor K,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const int height, const int width,
  const int pixel_border,
  const float z_eps,
  const float sigma_pixel, const float sigma_depth,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh) {

  CHECK_CONTIGUOUS(Twc);
  CHECK_CONTIGUOUS(Xs);
  CHECK_CONTIGUOUS(Cs);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(ii);
  CHECK_CONTIGUOUS(jj);
  CHECK_CONTIGUOUS(idx_ii2jj);
  CHECK_CONTIGUOUS(valid_match);
  CHECK_CONTIGUOUS(Q);

  // 调用 CUDA 实现的高斯牛顿校准调整
  return gauss_newton_calib_cuda(Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q,
      height, width, pixel_border, z_eps, sigma_pixel, sigma_depth, C_thresh, Q_thresh, max_iter, delta_thresh);
}

// 迭代投影
std::vector<torch::Tensor> iter_proj(
  torch::Tensor rays_img_with_grad, 
  torch::Tensor pts_3d_norm, 
  torch::Tensor p_init,
  const int max_iter,
  const float lambda_init,
  const float cost_thresh) {

  CHECK_CONTIGUOUS(rays_img_with_grad);
  CHECK_CONTIGUOUS(pts_3d_norm);
  CHECK_CONTIGUOUS(p_init);

  // 调用 CUDA 实现的迭代投影
  return iter_proj_cuda(rays_img_with_grad, pts_3d_norm, p_init, 
      max_iter, lambda_init, cost_thresh);
}

// 匹配点精炼
std::vector<torch::Tensor> refine_matches(
  torch::Tensor D11, 
  torch::Tensor D21,
  torch::Tensor p1, 
  const int window_size,
  const int dilation_max) {

  CHECK_CONTIGUOUS(D11);
  CHECK_CONTIGUOUS(D21);
  CHECK_CONTIGUOUS(p1);

  // 调用 CUDA 实现的匹配点精炼
  return refine_matches_cuda(D11, D21, p1, window_size, dilation_max);
}

// 使用 pybind11 将 C++ 函数绑定到 Python 模块中
// PYBIND11_MODULE 宏用于定义一个新的 Python 模块
// TORCH_EXTENSION_NAME 是模块的名称，m 是模块对象

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 将 C++ 函数 gauss_newton_points 绑定到 Python 函数 gauss_newton_points
  m.def("gauss_newton_points", &gauss_newton_points, "gauss_newton point adjustment");
  // 将 C++ 函数 gauss_newton_rays 绑定到 Python 函数 gauss_newton_rays
  m.def("gauss_newton_rays", &gauss_newton_rays, "gauss_newton ray adjustment");
  // 将 C++ 函数 gauss_newton_calib 绑定到 Python 函数 gauss_newton_calib
  m.def("gauss_newton_calib", &gauss_newton_calib, "gauss_newton calib adjustment");
  // 将 C++ 函数 iter_proj 绑定到 Python 函数 iter_proj
  m.def("iter_proj", &iter_proj, "iterative projection with generic camera");
  // 将 C++ 函数 refine_matches 绑定到 Python 函数 refine_matches
  m.def("refine_matches", &refine_matches, "refine match in local neighborhood");
}
