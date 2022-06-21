// uses tensorflow c bindings to create neural networks
// main is unittest until tests began to formalize as modules
// NOTE: one session is one graph
// NOTE: this library will add features as needed throughout my AI development

// TODO: create an API for Neural Networks that is very C like but ergonomic with lazy builders
#include <stdio.h>
#include <tf_attrtype.h>

#include "Model.c"
#include "Brains.h"

//PROTOTYPES
void Test_MatMul(Model* model);

void Test_Concat(Model* model);

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
	printf("Device list output: %d\n",
		TF_SessionListDevices(model->session, model->status));

	//UNITTESTS
	Test_Concat(model);
	Test_MatMul(model);
	//TODO: gradient test

	//CLOSE SESSION//
	delete_model(model);

	printf("Goodbye, World!\n");
	return 0;
}

void Test_Concat(Model* model)
{
	printf("\n\n");
	printf("Test_Concat\n");
	printf("\n");

	printf("\n\n");
	// BUILD GRAPH//
	const int64_t dims[] = { 0 };
	const int length = 1;
	TF_Operation* placeholder_op_g =
		placeholder(model, "c1", dims, length, TF_FLOAT);

	// create another placeholder op for the second tensor that will be
	// concatenated with the first
	const int64_t dims2[] = { 0 };
	const int length2 = 1;
	TF_Operation* placeholder_op2_g =
		placeholder(model, "c2", dims2, length2, TF_FLOAT);

	// add Concat op
	const int64_t dims3[] = { 0 };
	const int length3 = 0;
	TF_Operation* concat_dim_input_g =
		add_tensor_op(model, "Const", "c_dim", dims3, length3, TF_INT32, NULL, 0, NULL,
			0, sizeof(int));

	//get the output edge for concat operation in the graph
	TF_Output* concat_dim_input_out = malloc(sizeof(TF_Output));
	concat_dim_input_out[0] = (TF_Output){ concat_dim_input_g, 0 };

	// add placeholder ops as inputs to Concat op
	TF_Output* inputs = malloc(sizeof(TF_Output) * 2);
	inputs[0] = (TF_Output){ placeholder_op_g, 0 };
	inputs[1] = (TF_Output){ placeholder_op2_g, 0 };

	TF_Operation* test_op_g =
		add_tensor_op(model, "Concat", "c", dims, length, TF_FLOAT, concat_dim_input_out,
			1, inputs, 2, sizeof(float));

	// DATA INPUT //
	// INPUT TENSORS //
	printf("Setting up input tensors\n");
	int NumInputs = 2;
	TF_Output* Inputs = malloc(sizeof(TF_Output) * NumInputs);
	// insert placeholders into Inputs
	TF_Output first_input = { placeholder_op_g, 0 };
	Inputs[0] = first_input;
	TF_Output second_input = { placeholder_op2_g, 0 };
	Inputs[1] = second_input;

	// SETUP OUTPUT TENSOR //
	printf("Setting up output tensor\n");
	int NumOutputs = 1;
	TF_Output* Outputs = malloc(sizeof(TF_Output) * NumOutputs);
	TF_Output first_output = { test_op_g, 0 };
	Outputs[0] = first_output;

	// ALLOCATE DATA FOR INPUTS AND OUTPUTS //
	printf("Allocating data reference for inputs and outputs\n");
	TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumInputs);

	printf("building input tensors\n");
	float* data1 = malloc(sizeof(float));
	int64_t dims1[] = { 1 };
	TF_Tensor* float_tensor = new_tensor(TF_FLOAT, 1, dims1, data1, sizeof(float));
	float* data1_ptr = (float*)TF_TensorData(float_tensor);
	data1_ptr[0] = 1.0f;

	float* data2 = malloc(sizeof(float));
	int64_t dimsi2[] = { 1 };
	TF_Tensor* float_tensor2 = new_tensor(TF_FLOAT, 1, dimsi2, data2, sizeof(float));
	float* data2_ptr = (float*)TF_TensorData(float_tensor2);
	data2_ptr[0] = 2.0f;
	printf("Created input tensors\n");

	InputValues[0] = float_tensor;
	InputValues[1] = float_tensor2;

	// call TF_SessionRun with all NULL
	printf("TF_SessionRun success\n");

	TF_Tensor** OutputValues =
		(TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumOutputs);
	TF_SessionRun(model->session, NULL, Inputs, InputValues, NumInputs, Outputs,
		OutputValues, NumOutputs, 0, 0, NULL, model->status);
	printf("Verifying output\n");
	float* output_data = (float*)TF_TensorData(OutputValues[0]);
	printf("Output data: %f %f\n", output_data[0], output_data[1]);

	//run the session again with different inputs
	data1_ptr[0] = 3.0f;
	data2_ptr[0] = 4.0f;
	TF_SessionRun(model->session, NULL, Inputs, InputValues, NumInputs, Outputs,
		OutputValues, NumOutputs, 0, 0, NULL, model->status);
	float* last_output_data = (float*)TF_TensorData(OutputValues[0]);

	if (TF_GetCode(model->status) != TF_OK) {
		printf("TF: %s\n", TF_Message(model->status));
	}
	// verify output
	printf("Verifying output\n");
	printf("Output data: %f %f\n", output_data[0], output_data[1]);
	printf("Last output data: %f %f\n", last_output_data[0], last_output_data[1]);
	//delete all the tensors we allocated
	TF_DeleteTensor(float_tensor);
	TF_DeleteTensor(float_tensor2);
	TF_DeleteTensor(OutputValues[0]);
}

void Test_MatMul(Model* model)
{
	printf("\n\n");
	printf("Test_MatMul\n");
	printf("\n");


	//create an op for matrix multiply just like the above Concat code with two placeholder inputs
	const int64_t dims4[] = { 2, 2 };
	const int length4 = 2;
	TF_Operation* placeholder_op_g2 =
		placeholder(model, "c3", dims4, length4, TF_FLOAT);
	const int64_t dims5[] = { 2, 2 };
	const int length5 = 2;
	TF_Operation* placeholder_op_g3 =
		placeholder(model, "c4", dims5, length5, TF_FLOAT);

	//create a matrix multiply op
	TF_OperationDescription* test_op_g2 = TF_NewOperation(model->graph, "MatMul", "m");
	TF_SetAttrType(test_op_g2, "T", TF_FLOAT);
	TF_AddInput(test_op_g2, (TF_Output) { placeholder_op_g2, 0 });
	TF_AddInput(test_op_g2, (TF_Output) { placeholder_op_g3, 0 });
	TF_Operation* test_op_g3 = TF_FinishOperation(test_op_g2, model->status);
	printf("Created matrix multiply op\n");
	printTf(model);
	//build the inputs and outputs for MatrixMult Operation and run it
	//create input
	printf("Creating input tensors for matmul\n");
	int NumInputs_matmul = 2;
	TF_Output* Inputs_matmul = malloc(sizeof(TF_Output) * NumInputs_matmul);
	TF_Output first_input_matmul = { placeholder_op_g2, 0 };
	Inputs_matmul[0] = first_input_matmul;
	TF_Output second_input_matmul = { placeholder_op_g3, 0 };
	Inputs_matmul[1] = second_input_matmul;
	printf("Created input tensors for matmul\n");
	printf("creating outputs for matmul\n");
	int NumOutputs_matmul = 1;
	TF_Output* Outputs_matmul = malloc(sizeof(TF_Output) * NumOutputs_matmul);
	TF_Output first_output_matmul = { test_op_g3, 0 };
	Outputs_matmul[0] = first_output_matmul;
	printf("created outputs for matmul\n");
	printf("adding data to input tensors\n");
	TF_Tensor** InputValues_matmul = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumInputs_matmul);
	//generate data
	float* data1_matmul = malloc(sizeof(float));
	int64_t dims1_matmul[] = { 2, 2 };
	TF_Tensor* float_tensor_matmul = new_tensor(TF_FLOAT, 2, dims1_matmul, data1_matmul, sizeof(float) * 4);
	float* data1_ptr_matmul = (float*)TF_TensorData(float_tensor_matmul);
	data1_ptr_matmul[0] = 1.0f;
	data1_ptr_matmul[1] = 2.0f;
	data1_ptr_matmul[2] = 3.0f;
	data1_ptr_matmul[3] = 4.0f;
	printf("Created input tensors for matmul\n");

	float* data2_matmul = malloc(sizeof(float));
	int64_t dims2_matmul[] = { 2, 2 };
	TF_Tensor* float_tensor2_matmul = new_tensor(TF_FLOAT, 2, dims2_matmul, data2_matmul, sizeof(float) * 4);
	float* data2_ptr_matmul = (float*)TF_TensorData(float_tensor2_matmul);
	data2_ptr_matmul[0] = 3.0f;
	data2_ptr_matmul[1] = 4.0f;
	data2_ptr_matmul[2] = 5.0f;
	data2_ptr_matmul[3] = 6.0f;

	InputValues_matmul[0] = float_tensor_matmul;
	InputValues_matmul[1] = float_tensor2_matmul;
	printf("added data to input tensors\n");
	//create output values
	TF_Tensor** OutputValues_matmul =
		(TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumOutputs_matmul);

	printf("running matmul\n");
	TF_SessionRun(model->session, NULL, Inputs_matmul, InputValues_matmul, NumInputs_matmul, Outputs_matmul,
		OutputValues_matmul, NumOutputs_matmul, 0, 0, NULL, model->status);
	printf("ran matmul\n");
	//print outputs
	float* output_data_matmul = (float*)TF_TensorData(OutputValues_matmul[0]);
	printf("Output data: %f %f\n", output_data_matmul[0], output_data_matmul[1]);
	//print the other output
	printf("Output data: %f %f\n", output_data_matmul[2], output_data_matmul[3]);
}
