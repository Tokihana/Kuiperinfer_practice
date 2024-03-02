//
// Created by fss on 23-6-4.
//

#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
TEST(test_tensor_values, tensor_values1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Rand();
  f1.Show();

  LOG(INFO) << "Data in the first channel: " << f1.slice(0);
  LOG(INFO) << "Data in the (1,1,1): " << f1.at(1, 1, 1);
  f1.Fill(1.0f);
  ASSERT_EQ(1, f1.at(1, 1, 1));
  const auto& mat = f1.slice(0);
  arma::fmat mat_s = arma::fmat(3, 4, arma::fill::value(1.0f));
  ASSERT_TRUE(arma::approx_equal(mat, mat_s, "absdiff", 1e-6));
}