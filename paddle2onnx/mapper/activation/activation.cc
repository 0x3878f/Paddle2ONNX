// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle2onnx/mapper/activation/activation.h"
#include "paddle2onnx/mapper/exporter.h"

namespace paddle2onnx {

REGISTER_MAPPER(abs, ActivationMapper)
REGISTER_PIR_MAPPER(abs, ActivationMapper)
REGISTER_MAPPER(acos, ActivationMapper)
REGISTER_MAPPER(asin, ActivationMapper)
REGISTER_MAPPER(atan, ActivationMapper)
REGISTER_MAPPER(brelu, BReluMapper)
REGISTER_MAPPER(ceil, ActivationMapper)
REGISTER_PIR_MAPPER(ceil, ActivationMapper)
REGISTER_MAPPER(cos, ActivationMapper)
REGISTER_PIR_MAPPER(cos, ActivationMapper)
REGISTER_MAPPER(elu, EluMapper)
REGISTER_MAPPER(erf, ActivationMapper)
REGISTER_MAPPER(exp, ActivationMapper)
REGISTER_PIR_MAPPER(exp, ActivationMapper)
REGISTER_MAPPER(floor, ActivationMapper)
REGISTER_PIR_MAPPER(floor, ActivationMapper)
REGISTER_MAPPER(gelu, GeluMapper)
REGISTER_PIR_MAPPER(gelu, GeluMapper)
REGISTER_MAPPER(leaky_relu, LeakyReluMapper)
REGISTER_PIR_MAPPER(leaky_relu, LeakyReluMapper)
REGISTER_MAPPER(log, ActivationMapper)
REGISTER_PIR_MAPPER(log, ActivationMapper)
REGISTER_MAPPER(log10, Log10Mapper)
REGISTER_PIR_MAPPER(log10, Log10Mapper)
REGISTER_MAPPER(log1p, Log1PMapper)
REGISTER_PIR_MAPPER(log1p, Log1PMapper)
REGISTER_MAPPER(log2, Log2Mapper)
REGISTER_PIR_MAPPER(log2, Log2Mapper)
REGISTER_MAPPER(logsigmoid, LogSigmoidMapper)
REGISTER_MAPPER(log_softmax, LogSoftmaxMapper)
REGISTER_MAPPER(mish, MishMapper)
REGISTER_MAPPER(prelu, PReluMapper)
REGISTER_PIR_MAPPER(prelu, PReluMapper)
REGISTER_MAPPER(reciprocal, ActivationMapper)
REGISTER_MAPPER(relu, ActivationMapper)
REGISTER_PIR_MAPPER(relu, ActivationMapper)
REGISTER_MAPPER(round, ActivationMapper)
REGISTER_MAPPER(rsqrt, RsqrtMapper)
REGISTER_MAPPER(sel, ActivationMapper)
REGISTER_MAPPER(selu, SeluMapper)
REGISTER_MAPPER(silu, SiluMapper)
REGISTER_PIR_MAPPER(silu, SiluMapper)
REGISTER_MAPPER(sin, ActivationMapper)
REGISTER_PIR_MAPPER(sin, ActivationMapper)
REGISTER_MAPPER(size, SizeMapper)
REGISTER_PIR_MAPPER(size, SizeMapper)
REGISTER_MAPPER(softmax, SoftMaxMapper)
REGISTER_PIR_MAPPER(softmax, SoftMaxMapper)
REGISTER_MAPPER(softplus, ActivationMapper)
REGISTER_MAPPER(softshrink, SoftShrinkMapper)
REGISTER_MAPPER(softsign, ActivationMapper)
REGISTER_MAPPER(sqrt, ActivationMapper)
REGISTER_PIR_MAPPER(sqrt, ActivationMapper)
REGISTER_MAPPER(square, SquareMapper)
REGISTER_MAPPER(tan, ActivationMapper)
REGISTER_MAPPER(tanh, ActivationMapper)
REGISTER_PIR_MAPPER(tanh, ActivationMapper)
REGISTER_MAPPER(tanh_shrink, TanhShrinkMapper)
REGISTER_MAPPER(thresholded_relu, ThresholdedReluMapper)


int32_t ActivationMapper::GetMinOpsetVersion(bool verbose) {
  if (convert_pir_op_name(OpType()) == "softplus") {
    float beta = 0.0;
    float threshold = 20.0;
    GetAttr("beta", &beta);
    GetAttr("threshold", &threshold);
    if ((beta - 1.0) > 1e-06 || (beta - 1.0) < -1e-06 ||
        (threshold - 20.0) > 1e-06 || (threshold - 20.0) < -1e-06) {
      Error() << "Only support softplus with beta == 1.0 and threshold == 20.0."
              << std::endl;
      return -1;
    }
  }
  if (convert_pir_op_name(OpType()) == "round") {
    Logger(verbose, 11) << RequireOpset(11) << std::endl;
    return 11;
  }
  return 7;
}


void ActivationMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto iter = op_mapper_.find(convert_pir_op_name(OpType()));
  Assert(op_mapper_.end() != iter,
         "Cannot find " +
         convert_pir_op_name(OpType()) +
         " in activation op_mapper.");
  if (convert_pir_op_name(OpType()) == "erf") {
    auto input = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                   P2ODataType::FP32);
    auto output = helper_->MakeNode(iter->second, {input})->output(0);
    helper_->AutoCast(output, output_info[0].name, P2ODataType::FP32,
                      output_info[0].dtype);
  } else {
    helper_->MakeNode(iter->second, {input_info[0].name},
                      {output_info[0].name});
  }
}

int32_t PReluMapper::GetMinOpsetVersion(bool verbose) {
  auto input_info = GetInput("X");
  auto slope_info = GetInput("Alpha");
  if (input_info[0].Rank() != slope_info[0].Rank()) {
    if (slope_info[0].Rank() > 1) {
      Error()
          << "Only support rank of alpha <=1 while Rank(alpha) != Rank(input)."
          << std::endl;
      return -1;
    }
  }
  return 7;
}

void PReluMapper::Opset7() {
  auto input_info = GetInput("X");
  auto slope_info = GetInput("Alpha");
  auto output_info = GetOutput("Out");

  std::string slope_cast_name = slope_info[0].name;
  if (slope_info[0].dtype == P2ODataType::FP64) {
    slope_cast_name = helper_->AutoCast({slope_info[0].name}, P2ODataType::FP64,
                                        P2ODataType::FP32);
  }

  if (slope_info[0].Rank() != input_info[0].Rank()) {
    Assert(slope_info[0].Rank() <= 1,
           "Paddle2ONNX: Only support rank of alpha <= 1 while rank of alpha "
           "is not equal with rank of input for operator prelu.");
    Assert(
        input_info[0].Rank() > 1,
        "Paddle2ONNX: Rank of input should greater than 2 for operator prelu.");
    std::vector<int64_t> shape_value(input_info[0].Rank() - 1, 1);
    shape_value[0] = -1;
    slope_cast_name = helper_->Reshape(slope_cast_name, shape_value);
  }

  if (input_info[0].dtype == P2ODataType::FP64) {
    std::string x_cast_name = helper_->AutoCast(
        {input_info[0].name}, P2ODataType::FP64, P2ODataType::FP32);
    auto node = helper_->MakeNode("PRelu", {x_cast_name, slope_cast_name});
    helper_->AutoCast(node->output(0), {output_info[0].name}, P2ODataType::FP32,
                      P2ODataType::FP64);
  } else {
    helper_->MakeNode("PRelu", {input_info[0].name, slope_cast_name},
                      {output_info[0].name});
  }
}

void SeluMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto node =
      helper_->MakeNode("Selu", {input_info[0].name}, {output_info[0].name});
  AddAttribute(node, "alpha", alpha_);
  AddAttribute(node, "gamma", scale_);
}

void LeakyReluMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto node = helper_->MakeNode("LeakyRelu", {input_info[0].name},
                                {output_info[0].name});
  AddAttribute(node, "alpha", alpha_);
}

void GeluMapper::Opset9() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto input_onnx_dtype = GetOnnxDtype(input_info[0].dtype);
  double sqrt_2_value = 1.4142135623730951;
  double scale_value = 0.5;
  double const_1_value = 1.0;
  auto sqrt_2 =
      helper_->Constant({}, ONNX_NAMESPACE::TensorProto::FLOAT, sqrt_2_value);
  auto scale =
      helper_->Constant({}, ONNX_NAMESPACE::TensorProto::FLOAT, scale_value);
  auto const_1 =
      helper_->Constant({}, ONNX_NAMESPACE::TensorProto::FLOAT, const_1_value);

  auto input_name = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                      P2ODataType::FP32);

  // the computation formula follows
  // https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/gelu_cn.html#gelu
  auto erf0 = helper_->MakeNode("Div", {input_name, sqrt_2});
  auto erf1 = helper_->MakeNode("Erf", {erf0->output(0)});
  auto gelu0 = helper_->MakeNode("Add", {erf1->output(0), const_1});
  auto gelu1 = helper_->MakeNode("Mul", {input_name, gelu0->output(0)});

  if (input_info[0].dtype != P2ODataType::FP32) {
    auto out = helper_->MakeNode("Mul", {gelu1->output(0), scale});
    auto cast_out =
        helper_->MakeNode("Cast", {out->output(0)}, {output_info[0].name});
    AddAttribute(cast_out, "to", GetOnnxDtype(input_info[0].dtype));
  } else {
    helper_->MakeNode("Mul", {gelu1->output(0), scale}, {output_info[0].name});
  }
}

void SoftMaxMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  if (input_info[0].Rank() == 0) {
    auto unsqueeze = helper_->Unsqueeze(input_info[0].name, {0});
    auto node = helper_->MakeNode("Softmax", {unsqueeze});
    AddAttribute(node, "axis", static_cast<int64_t>(0));
    helper_->Squeeze(node->output(0), output_info[0].name, {0});
  } else {
    if (axis_ < 0) {
      axis_ = axis_ + output_info[0].Rank();
    }
    if (axis_ == output_info[0].Rank() - 1) {
      auto node = helper_->MakeNode("Softmax", {input_info[0].name},
                                    {output_info[0].name});
      AddAttribute(node, "axis", axis_);
    } else {
      std::vector<int64_t> perm = Arange(0, output_info[0].Rank());
      perm[output_info[0].Rank() - 1] = axis_;
      perm[axis_] = output_info[0].Rank() - 1;
      auto transpose_node =
          helper_->MakeNode("Transpose", {input_info[0].name});
      AddAttribute(transpose_node, "perm", perm);
      auto softmax_node =
          helper_->MakeNode("Softmax", {transpose_node->output(0)});
      int64_t axis_last = -1;
      AddAttribute(softmax_node, "axis", axis_last);
      auto transpose_node_last = helper_->MakeNode(
          "Transpose", {softmax_node->output(0)}, {output_info[0].name});
      AddAttribute(transpose_node_last, "perm", perm);
    }
  }
}

void SoftMaxMapper::Opset13() {
  int64_t axis;
  GetAttr("axis", &axis);
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  if (input_info[0].Rank() == 0) {
    auto unsqueeze = helper_->Unsqueeze(input_info[0].name, {0});
    auto node = helper_->MakeNode("Softmax", {unsqueeze});
    AddAttribute(node, "axis", static_cast<int64_t>(0));
    helper_->Squeeze(node->output(0), output_info[0].name, {0});
  } else {
    auto node = helper_->MakeNode("Softmax", {input_info[0].name},
                                  {output_info[0].name});
    AddAttribute(node, "axis", axis);
  }
}

void BReluMapper::Opset7() {
  auto x_info = GetInput("X");
  helper_->Clip(x_info[0].name, GetOutput("Out")[0].name, t_min_, t_max_,
                x_info[0].dtype);
}

void EluMapper::Opset7() {
  auto node = helper_->MakeNode("Elu", {GetInput("X")[0].name},
                                {GetOutput("Out")[0].name});
  AddAttribute(node, "alpha", alpha_);
}

int32_t MishMapper::GetMinOpsetVersion(bool verbose) {
  if (fabs(threshold_ - 20.0) > 1e-05) {
    Error() << "Only support threshold = 20.0." << std::endl;
    return -1;
  }
  return 7;
}

void MishMapper::Opset7() {
  auto input_info = GetInput("X");
  auto out_info = GetOutput("Out");
  auto input = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                 P2ODataType::FP32);
  auto softplus = helper_->MakeNode("Softplus", {input})->output(0);
  auto tanh = helper_->MakeNode("Tanh", {softplus})->output(0);
  auto output = helper_->MakeNode("Mul", {input, tanh})->output(0);
  helper_->AutoCast(output, out_info[0].name, P2ODataType::FP32,
                    out_info[0].dtype);
}

void SquareMapper::Opset7() {
  auto input_info = GetInput("X");
  helper_->MakeNode("Mul", {input_info[0].name, input_info[0].name},
                    {GetOutput("Out")[0].name});
}

void SoftShrinkMapper::Opset9() {
  auto node = helper_->MakeNode("Shrink", {GetInput("X")[0].name},
                                {GetOutput("Out")[0].name});
  AddAttribute(node, "lambd", lambda_);
  AddAttribute(node, "bias", lambda_);
}

void SizeMapper::Opset7() {
  auto out_info = GetOutput("Out");
  auto output =
      helper_->MakeNode("Size", {GetInput("Input")[0].name})->output(0);
  output = helper_->AutoCast(output, out_info[0].name, P2ODataType::INT64,
                             out_info[0].dtype);
}

void RsqrtMapper::Opset7() {
  auto output = helper_->MakeNode("Sqrt", {GetInput("X")[0].name})->output(0);
  helper_->MakeNode("Reciprocal", {output}, {GetOutput("Out")[0].name});
}

void TanhShrinkMapper::Opset7() {
  auto x_info = GetInput("X");
  auto tanh = helper_->MakeNode("Tanh", {x_info[0].name})->output(0);
  helper_->MakeNode("Sub", {x_info[0].name, tanh}, {GetOutput("Out")[0].name});
}

void LogSigmoidMapper::Opset7() {
  auto output =
      helper_->MakeNode("Sigmoid", {GetInput("X")[0].name})->output(0);
  helper_->MakeNode("Log", {output}, {GetOutput("Out")[0].name});
}

void LogSoftmaxMapper::Opset7() {
  auto input_info = GetInput("X");
  auto axis = axis_;
  if (input_info[0].Rank() == 0) {
    auto unsqueeze = helper_->Unsqueeze(input_info[0].name, {0});
    auto node = helper_->MakeNode("LogSoftmax", {unsqueeze});
    AddAttribute(node, "axis", static_cast<int64_t>(0));
    helper_->Squeeze(node->output(0), GetOutput("Out")[0].name, {0});
  } else {
    if (axis < 0) {
      axis += input_info[0].Rank();
    }
    if (axis == input_info[0].Rank() - 1) {
      auto node = helper_->MakeNode("LogSoftmax", {input_info[0].name},
                                    {GetOutput("Out")[0].name});
      AddAttribute(node, "axis", axis);
    } else {
      auto perm = Arange(0, input_info[0].Rank());
      perm[input_info[0].Rank() - 1] = axis;
      perm[axis] = input_info[0].Rank() - 1;
      auto output = helper_->Transpose(input_info[0].name, perm);
      auto node = helper_->MakeNode("LogSoftmax", {output});
      AddAttribute(node, "axis", int64_t(-1));
      helper_->Transpose(node->output(0), GetOutput("Out")[0].name, perm);
    }
  }
}

void ThresholdedReluMapper::Opset10() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  auto input = x_info[0].name;
  if (x_info[0].dtype != P2ODataType::FP32) {
    input = helper_->AutoCast(input, x_info[0].dtype, P2ODataType::FP32);
    auto node = helper_->MakeNode("ThresholdedRelu", {input});
    AddAttribute(node, "alpha", threshold_);
    helper_->AutoCast(node->output(0), out_info[0].name, P2ODataType::FP32,
                      out_info[0].dtype);
  } else {
    auto node =
        helper_->MakeNode("ThresholdedRelu", {input}, {out_info[0].name});
    AddAttribute(node, "alpha", threshold_);
  }
}

void Log1PMapper::Opset7() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  auto one = helper_->Constant({},
                               GetOnnxDtype(x_info[0].dtype),
                               static_cast<float>(1.0));
  auto input = helper_->MakeNode("Add", {x_info[0].name, one})->output(0);
  helper_->MakeNode("Log", {input}, {out_info[0].name});
}

void Log2Mapper::Opset7() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  double ln2 = 0.693147180559945309;
  auto ln2_tensor = helper_->Constant({}, GetOnnxDtype(x_info[0].dtype), ln2);
  auto output = helper_->MakeNode("Log", {x_info[0].name})->output(0);
  helper_->MakeNode("Div", {output, ln2_tensor}, {out_info[0].name});
}

void Log10Mapper::Opset7() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  double ln10 = 2.30258509299404568401;
  auto ln10_tensor = helper_->Constant({}, GetOnnxDtype(x_info[0].dtype), ln10);
  auto output = helper_->MakeNode("Log", {x_info[0].name})->output(0);
  helper_->MakeNode("Div", {output, ln10_tensor}, {out_info[0].name});
}

void SiluMapper::Opset7() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  auto out = helper_->MakeNode("Sigmoid", {x_info[0].name})->output(0);
  helper_->MakeNode("Mul", {x_info[0].name, out}, {out_info[0].name});
}
}  // namespace paddle2onnx
