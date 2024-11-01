// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle2onnx/mapper/exporter.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
namespace paddle2onnx {
void ModelExporter::ExportWhile(const PaddleParser& parser,
                                OnnxHelper* temp_helper, int32_t block_id,
                                int32_t op_id) {
  auto op = parser.GetOpDesc(block_id, op_id);
  auto x_info = parser.GetOpInput(block_id, op_id, "X");
  auto cond_info = parser.GetOpInput(block_id, op_id, "Condition");
  // auto out_info = parser.GetOpOutput(block_id, op_id, "Out");

  ONNX_NAMESPACE::GraphProto graph;
  /********************* Creat Body Gragh *********************/
  int32_t sub_block_idx = -1;
  for (size_t i = 0; i < op.attrs_size(); ++i) {
    if (op.attrs(i).name() == "sub_block") {
      sub_block_idx = op.attrs(i).block_idx();
      break;
    }
  }
  Assert(sub_block_idx > 0, "Cannot find sub_block in while operator.");

  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> parameters;
  std::vector<std::string> input_names;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> inputs;
  std::vector<std::string> output_names;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> outputs;

  auto iter_name = MapperHelper::Get()->GenName("loop.iter");
  TensorInfo iter_info(iter_name, std::vector<int64_t>(1, 1),
                       P2ODataType::INT64);
  inputs.push_back(std::move(MakeValueInfo(iter_info)));

  // Make cond
  input_names.push_back(cond_info[0].name);
  inputs.push_back(std::move(MakeValueInfo(cond_info[0])));
  outputs.push_back(std::move(std::move(MakeValueInfo(cond_info[0]))));

  // Make other inputs
  for (size_t i = 0; i < x_info.size(); ++i) {
    if (std::find(input_names.begin(), input_names.end(), x_info[i].name) !=
        input_names.end()) {
      continue;
    }

    if (!(x_info[i].is_tensor_array)) {
      // P2OLogger() << x_info[i].name << "is tensor array" << std::endl;
      inputs.push_back(std::move(MakeValueInfo(x_info[i])));
    }
    input_names.push_back(x_info[i].name);
    outputs.push_back(std::move(MakeValueInfo(x_info[i])));
  }

  graph = ExportBlock(parser, sub_block_idx, parameters, inputs, outputs, nullptr, true);

  /********************* Creat Body Gragh *********************/
  // Make Fake iter
  auto fake_iter = temp_helper->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                         std::vector<int64_t>(1, 1024));
  input_names.insert(input_names.begin(), fake_iter);
  for(int i=2;i<input_names.size();i++) {
    output_names.push_back(input_names[i]);
  }
  
  auto loop_node = temp_helper->MakeNode("Loop", input_names, output_names);
  AddAttribute(loop_node, "body", graph);
}

void ModelExporter::ExportWhile(PaddlePirParser& pir_parser,OnnxHelper* temp_helper,pir::Operation* op) {
  std::vector<TensorInfo> x_info;
  std::vector<TensorInfo> out_info;
  auto while_op = op->dyn_cast<paddle::dialect::WhileOp>();
  auto cond_info = pir_parser.GetTensorInfo(while_op.cond());
  std::vector<pir::detail::ValueImpl*> while_op_input_value_address;
  std::vector<pir::detail::ValueImpl*> while_op_input_arg_address;
  pir_parser.while_op_input_value_map.clear();
  //for while op inputs
  for(int index=1;index<while_op.num_operands();index++){
    const pir::Value& value = while_op.operand_source(index);
    while_op_input_value_address.push_back(&(*(value).impl()));
    x_info.push_back(pir_parser.GetTensorInfo(pir_parser.GetOpOutputName(value.defining_op()->result(0)),value.type()));
  }  
  //for while op args
  std::vector<pir::Value> args = while_op.block_args();
  for(int i = 0;i<args.size();i++){
    const pir::Value& value = args[i];
    while_op_input_arg_address.push_back(&(*(value.impl())));
  }
  //for the map between while op args and while op inputs
  for(int index=0;index<while_op_input_value_address.size();index++){
    pir_parser.while_op_input_value_map[while_op_input_arg_address[index]]=while_op_input_value_address[index];
  }

  pir_parser.sub_blocks_ops.clear();
  auto& body_block = while_op.body();
  for (auto& op : body_block.ops()) {
    if (op->name() != "builtin.parameter") {
      pir_parser.sub_blocks_ops.push_back(op);
    }
  }
  pir_parser.GetALLSubBlockOpOutputName(pir_parser.sub_blocks_ops);
  if (!pir_parser.sub_blocks_ops.empty()) {
    // get cf.yeild op input
    pir::Operation* cf_yield_op = pir_parser.sub_blocks_ops.back();
    for (auto oprand : cf_yield_op->operands()) {
      pir::Value value = oprand.source();
      auto info = pir_parser.GetSubBlockValueTensorInfo(value);
      out_info.push_back(info[0]);//和while op的cond的名字保持一致
    }
  } else {
    // sub_blocks_ops is empty
    PADDLE_ENFORCE_NE(pir_parser.sub_blocks_ops.size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The number of ops of a control flow sub-block "
                          "cannot be zero."));
  }
  // for(int i =1;i<out_info.size();i++){
  //   out_info[i].name="p2o.pd_op.while.0."+std::to_string(i-1);
  // }
  ONNX_NAMESPACE::GraphProto graph;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> parameters;
  std::vector<std::string> input_names;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> inputs;
  std::vector<std::string> output_names;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> outputs;
  auto iter_name = MapperHelper::Get()->GenName("loop.iter");
  TensorInfo iter_info(iter_name, std::vector<int64_t>(1, 1),
                      P2ODataType::INT64);
  //make inputs
  inputs.push_back(std::move(MakeValueInfo(iter_info)));
  input_names.push_back(cond_info[0].name);
  inputs.push_back(std::move(MakeValueInfo(cond_info[0])));
  //new
  outputs.push_back(std::move(std::move(MakeValueInfo(cond_info[0]))));


  for(size_t i=0;i<x_info.size();++i) {
    if (std::find(input_names.begin(), input_names.end(), x_info[i].name) !=
        input_names.end()) {
      continue;
    }
    inputs.push_back(std::move(MakeValueInfo(x_info[i])));
    input_names.push_back(x_info[i].name);
    //new
    outputs.push_back(std::move(MakeValueInfo(x_info[i])));
  }
  //make outputs
  for(size_t i=0;i<out_info.size();++i){
    // if(i==0){
    //   out_info[i].name = cond_info[0].name;
    // }else{
    //   out_info[i].name = x_info[i-1].name;
    // }
    if(i==0){
      // std::cout<<"cond name:"<< cond_info[0].name<<std::endl;
      out_info[i].name = "p2o.pd_op.less_equal.0.0";
    }else if(i==1){
      out_info[i].name="p2o.pd_op.while.0.0";
    }else if(i==2){
      out_info[i].name="p2o.pd_op.while.0.1";
    }
    outputs.push_back(std::move(MakeValueInfo(out_info[i])));
    output_names.push_back(out_info[i].name);
  }
  pir::Block* blockPtr = &body_block;
  graph = ExportBlock(pir_parser, blockPtr, parameters, inputs, outputs, true,true);

  /********************* Creat Body Gragh *********************/
  // Make Fake iter
  auto fake_iter = temp_helper->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                         std::vector<int64_t>(1, 1024));
  input_names.insert(input_names.begin(), fake_iter);
  for(int i=2;i<input_names.size();i++) {
    output_names.push_back(input_names[i]);
  }
  auto loop_node = temp_helper->MakeNode("Loop", input_names, output_names);
  AddAttribute(loop_node, "body", graph);
}
}  // namespace paddle2onnx