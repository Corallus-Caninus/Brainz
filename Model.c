#include <c_api.h>

// BEGIN CLASS//
struct Model {
	TF_Graph* graph;
	TF_Session* session;
	TF_SessionOptions* opts;
	TF_Status* status;
	TF_Output* inputs;
	TF_Output* outputs;
	// TODO: these must be done with AllocatTensor not NewTensor to ensure
	//       we own the closest data pointer to GPU mapped memory which is
	//       one of the primary goals of this framework.
	TF_Tensor* input_tensor;
	TF_Tensor* output_tensor;
};
// do the above as a typedef
typedef struct Model Model;
static void printTf(Model* model) {
	if (TF_GetCode(model->status) != TF_OK) {
		printf("%s\n", TF_Message(model->status));
	}
}

// TODO: want to return a struct that has function pointers that point to model
// internally and subtype function pointer so we dont pass model between every
// function (build your own C++ class)
static Model* build() {
	// initialize the graph etc.
	TF_Graph* graph = TF_NewGraph();
	TF_Status* status = TF_NewStatus();
	const TF_SessionOptions* opts = TF_NewSessionOptions();
	TF_Session* session = TF_NewSession(graph, opts, status);
	Model* model = malloc(sizeof(Model));
	model->graph = graph;
	model->session = session;
	model->opts = opts;
	model->status = status;
	printTf(model);
	// TODO: implement
	// model->inputs = malloc(sizeof(TF_Output));
	// model->outputs = malloc(sizeof(TF_Output));
	// model->input_tensor = malloc(sizeof(TF_Tensor));
	// model->output_tensor = malloc(sizeof(TF_Tensor));
	return model;
}

// TODO: find a way to subclass model so its stateful to the object using
// pointer to args struct?

/// <summary>
/// Basic Sequential Builder for a TensorFlow Model.
/// Builds from input to output
/// establishing edges eagerly. inserts the given operation in the current graph
/// and returns the operation's handle. attaches all given Outputs to the
/// operation's input edges. User is responsible for freeing the memory passed in.
/// </summary>
/// <param name="model"></param>
/// <param name="graph_op_name"></param>
/// the user defined name of the object in the graph
/// <param name="op_name"></param>
///  the name of the operation in Protobuff/Tensorflow
/// <param name="dims"></param>
///  the shape of the operations tensor
/// <param name="length"></param>
///  the size of the operations tensor
/// <param name="dtype"></param>
///  TF_Type for this operation
/// <param name="inputs"></param>
///  the operations to be assigned as inputs to this operation
/// <param name="num_inputs"></param>
///  the number of the aforementioned inputs
/// <param name="input_list"></param>
///  the list of inputs to be assigned to this operation
/// <param name="input_list_len"></param>
///  the length of the aforementioned list
/// <returns></returns>
static TF_Operation* add_tensor_op(Model* model, char* op_name_g, char* op_name,
	const int64_t* dims, int length, TF_DataType dtype,
	TF_Output* inputs, int num_inputs,
	TF_Output* input_list, int input_list_len,
	size_t type_size) {
	// add op to graph
	TF_OperationDescription* op =
		TF_NewOperation(model->graph, op_name_g, op_name);
	// add attr to op
	TF_SetAttrType(op, "dtype", dtype);
	printTf(model);
	TF_Tensor* value = TF_AllocateTensor(dtype, dims, length, type_size);
	// add shape
	//NOTE: for ops that dont have attributes these are ignored so these can be lazy NULL initialized builder
	TF_SetAttrShape(op, "shape", dims, length);
	printTf(model);

	// set the tensor to have value 1
	// TODO: does this only work because first value has no stride of TF_ length?
	*(int*)TF_TensorData(value) = 0;
	TF_SetAttrTensor(op, "value", value, model->status);

	// add the inputs to the graph
	for (int i = 0; i < num_inputs; i++) {
		printf("inp");
		TF_AddInput(op, inputs[i]);
		printTf(model);
	}
	// TODO: how to handle multiple input lists for some ops? for now just build
	// without helper function
	//  add the input_list
	if (input_list) TF_AddInputList(op, input_list, input_list_len);
	printTf(model);

	//  finish the op
	TF_Operation* operation = TF_FinishOperation(op, model->status);
	printTf(model);
	return operation;
}

// TODO: an entire malloc -> input placeholder abstraction helper function
// instead of or alongside this
/// <summary>
/// Creates a placeholder with the given name and shape.
/// </summary>
/// <param name="model"></param>
/// <param name="op_name"></param>
/// <param name="dims"></param>
/// <param name="length"></param>
/// <param name="dtype"></param>
/// <returns></returns>
static TF_Operation* placeholder(Model* model, char* op_name,
	const int64_t* dims, int length,
	TF_DataType dtype) {
	TF_OperationDescription* op =
		TF_NewOperation(model->graph, "Placeholder", op_name);
	printTf(model);
	TF_SetAttrType(op, "dtype", dtype);
	printTf(model);
	TF_SetAttrShape(op, "shape", dims, length);
	printTf(model);
	TF_Operation* operation = TF_FinishOperation(op, model->status);
	printTf(model);
	return operation;
}

// TODO: new_Constant //return a constant to be passed into an op
// TODO: new_Variable //return a variable to be passed into an op

/// <summary>
/// Retrieves the Operation from the graph by name and inserts the given input.
/// you shouldnt have to call this if building with the add_op sequential
/// builder correctly.
/// </summary>
/// <param name="model"></param>
/// <param name="graph_op_name"></param>
/// <param name="input"></param>
static void add_op_input(Model* model, char* graph_op_name, TF_Output input) {
	// add input to op
	TF_Operation* op = TF_GraphOperationByName(model->graph, graph_op_name);
	TF_AddInput(op, input);
}
// same as above but for input_list
/// <summary>
/// Retrieves the Operation from the graph by name and inserts the given list of
/// inputs.
/// </summary>
/// <param name="model"></param>
/// <param name="graph_op_name"></param>
/// <param name="inputs"></param>
/// <param name="num_inputs"></param>
static void add_op_input_list(Model* model, char* graph_op_name,
	TF_Input* inputs, int num_inputs) {
	// add input list to op
	TF_Operation* op = TF_GraphOperationByName(model->graph, graph_op_name);
	TF_AddInputList(op, inputs, num_inputs);
}
/// <summary>
/// Retrieves the Operation from the graph by name and inserts the given input.
/// </summary>
/// <param name="model"></param>
/// <param name="graph_op_name"></param>
/// <param name="index"></param>
/// <returns></returns>
static TF_Output get_op_output(Model* model, char* graph_op_name, int index) {
	// get output from op
	TF_Operation* op = TF_GraphOperationByName(model->graph, graph_op_name);
	TF_Output res = { op, index };
	return res;
}
/// <summary>
/// Retrieves the Operation from the graph by name and returns the output list.
/// </summary>
/// <param name="model"></param>
/// <param name="graph_op_name"></param>
/// <param name="start_index"></param>
/// <param name="end_index"></param>
/// <returns></returns>
static TF_Output* get_op_output_list(Model* model, char* graph_op_name,
	int start_index, int end_index) {
	// get output from op
	TF_Operation* op = TF_GraphOperationByName(model->graph, graph_op_name);
	TF_Output* res = malloc(sizeof(TF_Output) * (end_index - start_index + 1));
	for (int i = start_index; i <= end_index; i++) {
		TF_Output cur_output = { op, i };
		res[i - start_index] = cur_output;
	}
	return res;
}
//create a tensor and return it
//essentially we are making a function extraction of the following:
//int ndimsi1 = 1;
//int64_t dimsi1[] = { 1 };
//float* data1 = malloc(sizeof(float));
//*data1 = 1.0f;
//int ndata1 = sizeof(float);
//TF_Tensor* float_tensor = TF_AllocateTensor(TF_FLOAT, dimsi1, ndimsi1, ndata1);
////get the data pointer from the tensor
//float* data1_ptr = (float*)TF_TensorData(float_tensor);
//data1_ptr[0] = 1.0f;
static TF_Tensor* new_tensor(TF_DataType dtype, int ndims, int64_t* dims, float* data, int ndata) {
	TF_Tensor* tensor = TF_AllocateTensor(dtype, dims, ndims, ndata);
	float* data_ptr = (float*)TF_TensorData(tensor);
	for (int i = 0; i < ndata; i++) {
		data_ptr[i] = data[i];
	}
	return tensor;
}

/// <summary>
/// Cleanup routine for everything allocated by Model
/// if theres a leak something is missing here.
/// </summary>
/// <param name="model"></param>
static void delete_model(Model* model) {
	// delete model
	TF_DeleteStatus(model->status);
	TF_DeleteSession(model->session, model->status);
	TF_DeleteGraph(model->graph);
	// TODO: these point to graph worst case this is double-free
	if (model->inputs) free(model->inputs);
	if (model->outputs) free(model->outputs);
	TF_DeleteTensor(model->input_tensor);
	TF_DeleteTensor(model->output_tensor);
}
// END CLASS//
