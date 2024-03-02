//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_tensor, tensor_init1D) {
  using namespace kuiper_infer;
  Tensor<float> f1(4);
  f1.Fill(1.f);
  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor1D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t size = raw_shapes.at(0);
  LOG(INFO) << "data numbers: " << size;
  f1.Show();
}

TEST(test_tensor, tensor_init1DEQ) {
	using namespace kuiper_infer;
	Tensor<float> f1(4);
	f1.Fill(1.0f);
	const auto& shapes = f1.raw_shapes();
	std::vector<uint32_t> shapes_should_be = std::vector<uint32_t>{ 4 };
	ASSERT_EQ(shapes, shapes_should_be);
	const auto& data = f1.data();
	arma::fcube data_should_be = arma::fcube(1, 4, 1, arma::fill::value(1.0f));
	ASSERT_TRUE(approx_equal(data, data_should_be, "absdiff", 1e-6));
}

TEST(test_tensor, tensor_init2D) {
  using namespace kuiper_infer;
  Tensor<float> f1(4, 4);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor2D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_inti2DEQ) {
	using namespace kuiper_infer;
	Tensor<float> f1(4, 4);
	f1.Fill(1.0f);

	const auto& shapes = f1.raw_shapes();
	const auto& data = f1.data();
	std::vector<uint32_t> shapes_s = std::vector<uint32_t>{ 4, 4 };
	arma::fcube data_s = arma::fcube(4, 4, 1, arma::fill::value(1.0f));
	ASSERT_EQ(shapes, shapes_s);
	ASSERT_TRUE(arma::approx_equal(data, data_s, "absdiff", 1e-6));
}

TEST(test_tensor, tensor_init3D_3) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 3-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t channels = raw_shapes.at(0);
  const uint32_t rows = raw_shapes.at(1);
  const uint32_t cols = raw_shapes.at(2);

  LOG(INFO) << "data channels: " << channels;
  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_init3DEQ) {
	using namespace kuiper_infer;
	Tensor<float> f1(2, 3, 4);
	f1.Fill(1.0f);

	const auto& shapes = f1.raw_shapes();
	const auto& data = f1.data();
	std::vector<uint32_t> shapes_s = std::vector<uint32_t>{ 2, 3, 4 };
	arma::fcube data_s = arma::fcube(3, 4, 2, arma::fill::value(1.0f));
	ASSERT_EQ(shapes, shapes_s);
	ASSERT_TRUE(arma::approx_equal(data, data_s, "absdiff", 1e-6));
}

TEST(test_tensor, tensor_init3D_2) {
  using namespace kuiper_infer;
  Tensor<float> f1(1, 2, 3);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 2-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_init3D_1) {
  using namespace kuiper_infer;
  Tensor<float> f1(1, 1, 3);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 1-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t size = raw_shapes.at(0);

  LOG(INFO) << "data numbers: " << size;
  f1.Show();
}
