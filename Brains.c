// uses tensorflow c bindings to create neural networks
// main is unittest until tests began to formalize as modules
// NOTE: one session is one graph
// NOTE: this library will add features as needed throughout my AI development

// TODO: create an API for Neural Networks that is very C like but ergonomic
#include <stdio.h>
#include <tf_attrtype.h>

#include "Model.c"

// TODO: void close_model(dealloc), tensor* infer(tensor*), tensor*
// train(tensor*),
void NoOpDeallocator(void* data, size_t a, void* b) {}

// NOTE: see
// https://gist.github.com/asimshankar/7c9f8a9b04323e93bb217109da8c7ad2 for
// detailed example of all graph features in c api
int main(int argc, char** argv) {
  printf("Hello, World!\n");
  // INIT SESSION//
  Model* model = build();

  // print the device currently running
  printf("Device list output: %d\n",
         TF_SessionListDevices(model->session, model->status));

  // BUILD GRAPH//
  const int64_t dims[] = {0};
  const int length = 1;
  TF_Operation* placeholder_op_g =
      Placeholder(model, "c1", dims, length, TF_FLOAT);

  // create another placeholder op for the second tensor that will be
  // concatenated with the first
  const int64_t dims2[] = {0};
  const int length2 = 1;
  TF_Operation* Placeholder_op2_g =
      placeholder(model, "c2", dims2, length2, TF_FLOAT);

  // add Concat op
  const int64_t dims3[] = {0};
  const int length3 = 0;
  TF_Operation* concat_dim_input_g =
      add_op(model, "Const", "c_dim", dims3, length3, TF_INT32, NULL, 0, NULL,
             0, sizeof(int));

  // TF_Output concat_dim_input_out = {concat_dim_input_g, 0};
  TF_Output* concat_dim_input_out = malloc(sizeof(TF_Output));
  concat_dim_input_out[0] = (TF_Output){concat_dim_input_g, 0};
  // TF_AddInput(test_op, concat_dim_input_out);

  // add placeholder ops as inputs to Concat op
  TF_Output* inputs = malloc(sizeof(TF_Output) * 2);
  inputs[0] = (TF_Output){placeholder_op_g, 0};
  inputs[1] = (TF_Output){placeholder_op2_g, 0};

  TF_Operation* test_op_g =
      add_op(model, "Concat", "c", dims, length, TF_FLOAT, concat_dim_input_out,
             1, inputs, 2, sizeof(float));

  // DATA INPUT //
  // TODO: prefer to map to an array in c with TF_AllocateTensor so we can
  // update rapidly for sessions. then refactor into helper allocation init
  // function and corresponding helper deallocation function (free and TF) SETUP
  // INPUT TENSORS //
  printf("Setting up input tensors\n");
  int NumInputs = 2;
  TF_Output* Inputs = malloc(sizeof(TF_Output) * NumInputs);
  // insert placeholders into Inputs
  // Inputs[0] = {placeholder_op_g, 0};
  TF_Output first_input = {placeholder_op_g, 0};
  Inputs[0] = first_input;
  TF_Output second_input = {placeholder_op2_g, 0};
  Inputs[1] = second_input;

  // SETUP OUTPUT TENSOR //
  printf("Setting up output tensor\n");
  int NumOutputs = 1;
  TF_Output* Outputs = malloc(sizeof(TF_Output) * NumOutputs);
  TF_Output first_output = {test_op_g, 0};
  Outputs[0] = first_output;

  // ALLOCATE DATA FOR INPUTS AND OUTPUTS //
  printf("Allocating data for inputs and outputs\n");
  TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumInputs);
  printf("Allocated data for inputs and outputs\n");
  int ndimsi1 = 1;
  int64_t dimsi1[] = {1};
  float data1[] = {1.0f};
  int ndata1 = sizeof(float);
  TF_Tensor* float_tensor = TF_NewTensor(TF_FLOAT, dimsi1, ndimsi1, data1,
                                         ndata1, &NoOpDeallocator, NULL);

  int ndimsi2 = 1;
  int64_t dimsi2[] = {1};
  float data2[] = {2.0f};
  // int ndata = sizeof(int64_t);
  int ndata2 = sizeof(float);
  TF_Tensor* float_tensor2 = TF_NewTensor(TF_FLOAT, dimsi2, ndimsi2, data2,
                                          ndata2, &NoOpDeallocator, NULL);
  printf("Created input tensors\n");

  InputValues[0] = float_tensor;
  InputValues[1] = float_tensor2;

  // call TF_SessionRun with all NULL
  if (TF_GetCode(model->status) != TF_OK) {
    fprintf(stderr, "TF_SessionRun error: %s\n", TF_Message(model->status));
    return 1;
  }
  printf("TF_SessionRun success\n");

  TF_Tensor** OutputValues =
      (TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumOutputs);
  TF_SessionRun(model->session, NULL, Inputs, InputValues, NumInputs, Outputs,
                OutputValues, NumOutputs, 0, 0, NULL, model->status);

  if (TF_GetCode(model->status) != TF_OK) {
    printf("TF: %s\n", TF_Message(model->status));
  }
  // verify output
  printf("Verifying output\n");
  float* output_data = (float*)TF_TensorData(OutputValues[0]);
  printf("Output data: %f %f\n", output_data[0], output_data[1]);

  printf("Goodbye, World!\n");
  delete_model(model);
  return 0;
}
