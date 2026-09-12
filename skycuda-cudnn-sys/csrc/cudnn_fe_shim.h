/*
 * C ABI wrapper over cudnn-frontend C++.
 * Error handling - int32_t status is returned, nonzero indicates error,
 * error message available from cudnn_fe_last_error_message().
 * C ABI shim over NVIDIA cudnn-frontend's graph::Graph (Graph API).
 */
#ifndef CUDNN_FE_SHIM_H
#define CUDNN_FE_SHIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef enum cudnnFeStatus {
    CUDNN_FE_OK = 0,
    CUDNN_FE_ATTRIBUTE_NOT_SET = 1,
    CUDNN_FE_SHAPE_DEDUCTION_FAILED = 2,
    CUDNN_FE_INVALID_TENSOR_NAME = 3,
    CUDNN_FE_INVALID_VARIANT_PACK = 4,
    CUDNN_FE_GRAPH_NOT_SUPPORTED = 5,
    CUDNN_FE_GRAPH_EXECUTION_PLAN_CREATION_FAILED = 6,
    CUDNN_FE_GRAPH_EXECUTION_FAILED = 7,
    CUDNN_FE_HEURISTIC_QUERY_FAILED = 8,
    CUDNN_FE_UNSUPPORTED_GRAPH_FORMAT = 9,
    CUDNN_FE_CUDA_API_FAILED = 10,
    CUDNN_FE_CUDNN_BACKEND_API_FAILED = 11,
    CUDNN_FE_INVALID_CUDA_DEVICE = 12,
    CUDNN_FE_HANDLE_ERROR = 13,
    CUDNN_FE_INVALID_VALUE = 14,
    CUDNN_FE_NVRTC_COMPILATION_FAILED = 15,
    CUDNN_FE_SHIM_EXCEPTION = 1000
} cudnnFeStatus;

typedef enum cudnnFeDataType {
    CUDNN_FE_DATA_TYPE_NOT_SET = 0,
    CUDNN_FE_DATA_TYPE_FLOAT,
    CUDNN_FE_DATA_TYPE_DOUBLE,
    CUDNN_FE_DATA_TYPE_HALF,
    CUDNN_FE_DATA_TYPE_INT8,
    CUDNN_FE_DATA_TYPE_INT32,
    CUDNN_FE_DATA_TYPE_INT8x4,
    CUDNN_FE_DATA_TYPE_UINT8,
    CUDNN_FE_DATA_TYPE_UINT8x4,
    CUDNN_FE_DATA_TYPE_INT8x32,
    CUDNN_FE_DATA_TYPE_BFLOAT16,
    CUDNN_FE_DATA_TYPE_INT64,
    CUDNN_FE_DATA_TYPE_BOOLEAN,
    CUDNN_FE_DATA_TYPE_BYTE_BOOLEAN,
    CUDNN_FE_DATA_TYPE_FP8_E4M3,
    CUDNN_FE_DATA_TYPE_FP8_E5M2,
    CUDNN_FE_DATA_TYPE_FAST_FLOAT_FOR_FP8,
    CUDNN_FE_DATA_TYPE_FP8_E8M0,
    CUDNN_FE_DATA_TYPE_FP8_E5M3,
    CUDNN_FE_DATA_TYPE_FP4_E2M1,
    CUDNN_FE_DATA_TYPE_INT4,
    CUDNN_FE_DATA_TYPE_COMPLEX_FP32,
    CUDNN_FE_DATA_TYPE_COMPLEX_FP64
} cudnnFeDataType;

typedef enum cudnnFeHeurMode {
    CUDNN_FE_HEUR_MODE_A = 0,
    CUDNN_FE_HEUR_MODE_B = 1,
    CUDNN_FE_HEUR_MODE_FALLBACK = 2
    // We don't expose OPENSOURCE experimental mode.
} cudnnFeHeurMode;

typedef enum cudnnFeBuildPlanPolicy {
    CUDNN_FE_BUILD_PLAN_POLICY_HEURISTICS_CHOICE = 0,
    CUDNN_FE_BUILD_PLAN_POLICY_ALL = 1
} cudnnFeBuildPlanPolicy;

typedef enum cudnnFePointwiseMode {
    CUDNN_FE_POINTWISE_NOT_SET = 0,
    CUDNN_FE_POINTWISE_ADD,
    CUDNN_FE_POINTWISE_MUL,
    CUDNN_FE_POINTWISE_SQRT,
    CUDNN_FE_POINTWISE_MAX,
    CUDNN_FE_POINTWISE_MIN,
    CUDNN_FE_POINTWISE_RELU_FWD,
    CUDNN_FE_POINTWISE_TANH_FWD,
    CUDNN_FE_POINTWISE_SIGMOID_FWD,
    CUDNN_FE_POINTWISE_ELU_FWD,
    CUDNN_FE_POINTWISE_GELU_FWD,
    CUDNN_FE_POINTWISE_SOFTPLUS_FWD,
    CUDNN_FE_POINTWISE_SWISH_FWD,
    CUDNN_FE_POINTWISE_RELU_BWD,
    CUDNN_FE_POINTWISE_TANH_BWD,
    CUDNN_FE_POINTWISE_SIGMOID_BWD,
    CUDNN_FE_POINTWISE_ELU_BWD,
    CUDNN_FE_POINTWISE_GELU_BWD,
    CUDNN_FE_POINTWISE_SOFTPLUS_BWD,
    CUDNN_FE_POINTWISE_SWISH_BWD,
    CUDNN_FE_POINTWISE_ERF,
    CUDNN_FE_POINTWISE_IDENTITY,
    CUDNN_FE_POINTWISE_GELU_APPROX_TANH_BWD,
    CUDNN_FE_POINTWISE_GELU_APPROX_TANH_FWD,
    CUDNN_FE_POINTWISE_GEN_INDEX,
    CUDNN_FE_POINTWISE_BINARY_SELECT,
    CUDNN_FE_POINTWISE_EXP,
    CUDNN_FE_POINTWISE_LOG,
    CUDNN_FE_POINTWISE_NEG,
    CUDNN_FE_POINTWISE_MOD,
    CUDNN_FE_POINTWISE_POW,
    CUDNN_FE_POINTWISE_ABS,
    CUDNN_FE_POINTWISE_CEIL,
    CUDNN_FE_POINTWISE_COS,
    CUDNN_FE_POINTWISE_FLOOR,
    CUDNN_FE_POINTWISE_RSQRT,
    CUDNN_FE_POINTWISE_SIN,
    CUDNN_FE_POINTWISE_LOGICAL_NOT,
    CUDNN_FE_POINTWISE_TAN,
    CUDNN_FE_POINTWISE_SUB,
    CUDNN_FE_POINTWISE_ADD_SQUARE,
    CUDNN_FE_POINTWISE_DIV,
    CUDNN_FE_POINTWISE_CMP_EQ,
    CUDNN_FE_POINTWISE_CMP_NEQ,
    CUDNN_FE_POINTWISE_CMP_GT,
    CUDNN_FE_POINTWISE_CMP_GE,
    CUDNN_FE_POINTWISE_CMP_LT,
    CUDNN_FE_POINTWISE_CMP_LE,
    CUDNN_FE_POINTWISE_LOGICAL_AND,
    CUDNN_FE_POINTWISE_LOGICAL_OR,
    CUDNN_FE_POINTWISE_RECIPROCAL
} cudnnFePointwiseMode;

typedef enum cudnnFeReductionMode {
    CUDNN_FE_REDUCTION_NOT_SET = 0,
    CUDNN_FE_REDUCTION_ADD,
    CUDNN_FE_REDUCTION_MUL,
    CUDNN_FE_REDUCTION_MIN,
    CUDNN_FE_REDUCTION_MAX,
    CUDNN_FE_REDUCTION_AMAX,
    CUDNN_FE_REDUCTION_AVG,
    CUDNN_FE_REDUCTION_NORM1,
    CUDNN_FE_REDUCTION_NORM2,
    CUDNN_FE_REDUCTION_MUL_NO_ZEROS
} cudnnFeReductionMode;

typedef enum cudnnFeNormFwdPhase {
    CUDNN_FE_NORM_FWD_NOT_SET = 0,
    CUDNN_FE_NORM_FWD_INFERENCE,
    CUDNN_FE_NORM_FWD_TRAINING
} cudnnFeNormFwdPhase;

typedef enum cudnnFeConvolutionMode {
    CUDNN_FE_CONV_MODE_NOT_SET = 0,
    CUDNN_FE_CONV_MODE_CONVOLUTION,
    CUDNN_FE_CONV_MODE_CROSS_CORRELATION
} cudnnFeConvolutionMode;

typedef enum cudnnFeResampleMode {
    CUDNN_FE_RESAMPLE_NOT_SET = 0,
    CUDNN_FE_RESAMPLE_AVGPOOL_EXCLUDE_PADDING,
    CUDNN_FE_RESAMPLE_AVGPOOL_INCLUDE_PADDING,
    CUDNN_FE_RESAMPLE_BILINEAR,
    CUDNN_FE_RESAMPLE_NEAREST,
    CUDNN_FE_RESAMPLE_MAXPOOL
} cudnnFeResampleMode;

typedef enum cudnnFePaddingMode {
    CUDNN_FE_PADDING_NOT_SET = 0,
    CUDNN_FE_PADDING_EDGE_VAL_PAD,
    CUDNN_FE_PADDING_NEG_INF_PAD,
    CUDNN_FE_PADDING_ZERO_PAD
} cudnnFePaddingMode;

typedef enum cudnnFeReshapeMode {
    CUDNN_FE_RESHAPE_NOT_SET = 0,
    CUDNN_FE_RESHAPE_VIEW_ONLY,
    CUDNN_FE_RESHAPE_LOGICAL
} cudnnFeReshapeMode;

typedef enum cudnnFeTensorReordering {
    CUDNN_FE_TENSOR_REORDERING_NONE = 0,
    CUDNN_FE_TENSOR_REORDERING_INT8x32,
    CUDNN_FE_TENSOR_REORDERING_F16x16,
    CUDNN_FE_TENSOR_REORDERING_F8_128x4
} cudnnFeTensorReordering;

typedef enum cudnnFeDiagonalAlignment {
    CUDNN_FE_DIAGONAL_ALIGNMENT_TOP_LEFT = 0,
    CUDNN_FE_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT = 1
} cudnnFeDiagonalAlignment;

typedef enum cudnnFeRngDistribution {
    CUDNN_FE_RNG_DIST_NOT_SET = 0,
    CUDNN_FE_RNG_DIST_BERNOULLI,
    CUDNN_FE_RNG_DIST_UNIFORM,
    CUDNN_FE_RNG_DIST_NORMAL
} cudnnFeRngDistribution;

typedef enum cudnnFeMoeGroupedMatmulMode {
    CUDNN_FE_MOE_GMM_NOT_SET = 0,
    CUDNN_FE_MOE_GMM_NONE,
    CUDNN_FE_MOE_GMM_GATHER,
    CUDNN_FE_MOE_GMM_SCATTER
} cudnnFeMoeGroupedMatmulMode;

typedef enum cudnnFeAttentionImplementation {
    CUDNN_FE_ATTENTION_IMPL_AUTO = 0,
    CUDNN_FE_ATTENTION_IMPL_COMPOSITE,
    CUDNN_FE_ATTENTION_IMPL_UNIFIED
} cudnnFeAttentionImplementation;

typedef struct cudnnFeGraph cudnnFeGraph;
typedef struct cudnnFeTensor cudnnFeTensor;
typedef struct cudnnFeHandle cudnnFeHandle;
typedef struct cudnnFeKernelCache cudnnFeKernelCache;

const char* cudnn_fe_last_error_message(void);

/* e.g. 91500 */
int64_t cudnn_fe_backend_version(void);

/* Explicit initialization with dlopen. This is not thread-safe. */
int32_t cudnn_fe_load_library(const char* path);

/* cuda_stream is a CUstream. */
int32_t cudnn_fe_handle_create(void* cuda_stream, cudnnFeHandle** out);
int32_t cudnn_fe_handle_set_stream(cudnnFeHandle*, void* cuda_stream);
void    cudnn_fe_handle_destroy(cudnnFeHandle*);

int32_t cudnn_fe_kernel_cache_create(cudnnFeKernelCache** out);
void    cudnn_fe_kernel_cache_destroy(cudnnFeKernelCache*);

int32_t cudnn_fe_graph_create(cudnnFeGraph** out);
void    cudnn_fe_graph_destroy(cudnnFeGraph*);

int32_t cudnn_fe_graph_set_io_data_type(cudnnFeGraph*, cudnnFeDataType);
int32_t cudnn_fe_graph_set_intermediate_data_type(cudnnFeGraph*, cudnnFeDataType);
int32_t cudnn_fe_graph_set_compute_data_type(cudnnFeGraph*, cudnnFeDataType);
int32_t cudnn_fe_graph_set_name(cudnnFeGraph*, const char* name);
int32_t cudnn_fe_graph_set_dynamic_shape_enabled(cudnnFeGraph*, int32_t enabled);
/* Must be called before build. Requires cuDNN backend >= 9.21.0 at execution time. */
int32_t cudnn_fe_graph_set_override_shape_enabled(cudnnFeGraph*, int32_t enabled);
int32_t cudnn_fe_graph_set_sm_count(cudnnFeGraph*, int32_t sm_count);
int32_t cudnn_fe_graph_set_sm_version(cudnnFeGraph*, int32_t sm_version);
int32_t cudnn_fe_graph_set_kernel_cache(cudnnFeGraph*, cudnnFeKernelCache*);

typedef struct cudnnFeTensorDesc {
    const int64_t* dim;       /* required */
    size_t         dim_len;
    const int64_t* stride;    /* required */
    size_t         stride_len;
    cudnnFeDataType data_type;
    int32_t is_virtual;
    int32_t is_pass_by_value;
    int32_t has_uid;
    int64_t uid;              /* used only when has_uid != 0 */
    int64_t alignment;        /* 0 => leave default */
    cudnnFeTensorReordering reordering;
    const cudnnFeTensor* ragged_offset; /* nullable */
    int64_t ragged_offset_multiplier;   /* 0 => leave default */
    const char* name;         /* nullable */
} cudnnFeTensorDesc;

int32_t cudnn_fe_graph_tensor(cudnnFeGraph*, const cudnnFeTensorDesc*, cudnnFeTensor** out);
int32_t cudnn_fe_graph_tensor_like(cudnnFeGraph*, const cudnnFeTensor* like,
                                   const char* name, cudnnFeTensor** out);

int32_t cudnn_fe_tensor_get_uid(const cudnnFeTensor*, int64_t* out);
int32_t cudnn_fe_tensor_set_uid(cudnnFeTensor*, int64_t uid);
/* Copies up to *len dims into `out` */
int32_t cudnn_fe_tensor_get_dim(const cudnnFeTensor*, int64_t* out, size_t* len);
int32_t cudnn_fe_tensor_get_stride(const cudnnFeTensor*, int64_t* out, size_t* len);
int32_t cudnn_fe_tensor_set_output(cudnnFeTensor*, int32_t is_output);
int32_t cudnn_fe_tensor_get_data_type(const cudnnFeTensor*, cudnnFeDataType* out);
int32_t cudnn_fe_tensor_get_is_virtual(const cudnnFeTensor*, int32_t* out);
int32_t cudnn_fe_tensor_get_is_pass_by_value(const cudnnFeTensor*, int32_t* out);
int32_t cudnn_fe_tensor_get_reordering(const cudnnFeTensor*, cudnnFeTensorReordering* out);
/* Element size in bits as known to cudnn_frontend; 0 if unknown. */
int32_t cudnn_fe_data_type_size_bits(cudnnFeDataType, int64_t* out);
int32_t cudnn_fe_tensor_set_data_type(cudnnFeTensor*, cudnnFeDataType);
int32_t cudnn_fe_tensor_set_dim(cudnnFeTensor*, const int64_t* dim, size_t len);
int32_t cudnn_fe_tensor_set_stride(cudnnFeTensor*, const int64_t* stride, size_t len);
/* `offset` must be a tensor of the same graph. multiplier 0 => leave default. */
int32_t cudnn_fe_tensor_set_ragged_offset(cudnnFeGraph*, cudnnFeTensor*, const cudnnFeTensor* offset,
                                          int64_t multiplier);
int32_t cudnn_fe_tensor_get_has_ragged_offset(const cudnnFeTensor*, int32_t* out);

typedef struct cudnnFeMatmulAttrs {
    cudnnFeDataType compute_data_type;
    int32_t has_padding;
    double  padding;
    const char* name;
} cudnnFeMatmulAttrs;
int32_t cudnn_fe_graph_matmul(cudnnFeGraph*, const cudnnFeTensor* a, const cudnnFeTensor* b,
                              const cudnnFeMatmulAttrs*, cudnnFeTensor** c);

typedef struct cudnnFePointwiseAttrs {
    cudnnFePointwiseMode mode;
    cudnnFeDataType compute_data_type;
    int32_t has_axis;              int64_t axis;
    int32_t has_relu_lower_clip_slope; float relu_lower_clip_slope;
    int32_t has_relu_lower_clip;   float relu_lower_clip;
    int32_t has_relu_upper_clip;   float relu_upper_clip;
    int32_t has_swish_beta;        float swish_beta;
    int32_t has_elu_alpha;         float elu_alpha;
    int32_t has_softplus_beta;     float softplus_beta;
    const char* name;
} cudnnFePointwiseAttrs;
int32_t cudnn_fe_graph_pointwise_unary(cudnnFeGraph*, const cudnnFeTensor* a,
                                       const cudnnFePointwiseAttrs*, cudnnFeTensor** out);
int32_t cudnn_fe_graph_pointwise_binary(cudnnFeGraph*, const cudnnFeTensor* a, const cudnnFeTensor* b,
                                        const cudnnFePointwiseAttrs*, cudnnFeTensor** out);
int32_t cudnn_fe_graph_pointwise_ternary(cudnnFeGraph*, const cudnnFeTensor* a, const cudnnFeTensor* b,
                                         const cudnnFeTensor* c, const cudnnFePointwiseAttrs*,
                                         cudnnFeTensor** out);

typedef struct cudnnFeReductionAttrs {
    cudnnFeReductionMode mode;
    cudnnFeDataType compute_data_type;
    int32_t has_is_deterministic; int32_t is_deterministic;
    const char* name;
} cudnnFeReductionAttrs;
int32_t cudnn_fe_graph_reduction(cudnnFeGraph*, const cudnnFeTensor* a,
                                 const cudnnFeReductionAttrs*, cudnnFeTensor** out);

typedef struct cudnnFeConvAttrs {
    cudnnFeDataType compute_data_type;
    cudnnFeConvolutionMode conv_mode;
    const int64_t* pre_padding;  size_t pre_padding_len;
    const int64_t* post_padding; size_t post_padding_len;
    const int64_t* stride;       size_t stride_len;
    const int64_t* dilation;     size_t dilation_len;
    const char* name;
} cudnnFeConvAttrs;
int32_t cudnn_fe_graph_conv_fprop(cudnnFeGraph*, const cudnnFeTensor* x, const cudnnFeTensor* w,
                                  const cudnnFeConvAttrs*, cudnnFeTensor** y);
int32_t cudnn_fe_graph_conv_dgrad(cudnnFeGraph*, const cudnnFeTensor* dy, const cudnnFeTensor* w,
                                  const cudnnFeConvAttrs*, cudnnFeTensor** dx);
int32_t cudnn_fe_graph_conv_wgrad(cudnnFeGraph*, const cudnnFeTensor* x, const cudnnFeTensor* dy,
                                  const cudnnFeConvAttrs*, cudnnFeTensor** dw);

typedef struct cudnnFeReshapeAttrs {
    cudnnFeReshapeMode mode;         /* NOT_SET => leave default */
    const int64_t* dim;    size_t dim_len;    /* optional; len 0 => unset */
    const int64_t* stride; size_t stride_len; /* optional; len 0 => unset */
    const char* name;
} cudnnFeReshapeAttrs;
int32_t cudnn_fe_graph_reshape(cudnnFeGraph*, const cudnnFeTensor* x,
                               const cudnnFeReshapeAttrs*, cudnnFeTensor** y);

typedef struct cudnnFeTransposeAttrs {
    cudnnFeDataType compute_data_type;
    const int64_t* permutation; size_t permutation_len;
    const char* name;
} cudnnFeTransposeAttrs;
int32_t cudnn_fe_graph_transpose(cudnnFeGraph*, const cudnnFeTensor* x,
                                 const cudnnFeTransposeAttrs*, cudnnFeTensor** y);

typedef struct cudnnFeSliceAttrs {
    cudnnFeDataType compute_data_type;
    /* slices as flat [start0,end0, start1,end1, ...]; slices_len = 2*ndims */
    const int64_t* slices; size_t slices_len;
    const char* name;
} cudnnFeSliceAttrs;
int32_t cudnn_fe_graph_slice(cudnnFeGraph*, const cudnnFeTensor* x,
                             const cudnnFeSliceAttrs*, cudnnFeTensor** y);

typedef struct cudnnFeConcatenateAttrs {
    cudnnFeDataType compute_data_type;
    int64_t axis;
    int32_t has_in_place_index; int64_t in_place_index;
    const char* name;
} cudnnFeConcatenateAttrs;
int32_t cudnn_fe_graph_concatenate(cudnnFeGraph*, const cudnnFeTensor* const* inputs, size_t n_inputs,
                                   const cudnnFeConcatenateAttrs*, cudnnFeTensor** y);

typedef struct cudnnFeResampleAttrs {
    cudnnFeDataType compute_data_type;
    cudnnFeResampleMode resample_mode;
    cudnnFePaddingMode  padding_mode;
    int32_t has_is_inference; int32_t is_inference;
    int32_t generate_index;
    const int64_t* window;       size_t window_len;
    const int64_t* stride;       size_t stride_len;
    const int64_t* pre_padding;  size_t pre_padding_len;
    const int64_t* post_padding; size_t post_padding_len;
    const char* name;
} cudnnFeResampleAttrs;
/* out: y, index (index may be NULL when generate_index == 0) */
int32_t cudnn_fe_graph_resample(cudnnFeGraph*, const cudnnFeTensor* x,
                                const cudnnFeResampleAttrs*, cudnnFeTensor** y, cudnnFeTensor** index);

typedef struct cudnnFeRngAttrs {
    cudnnFeDataType compute_data_type;
    cudnnFeRngDistribution distribution;
    int32_t has_seed;   int64_t seed;
    int32_t has_prob;   double bernoulli_probability;
    const int64_t* dim;    size_t dim_len;    /* optional */
    const int64_t* stride; size_t stride_len; /* optional */
    const char* name;
} cudnnFeRngAttrs;
/* seed/offset tensors are optional (may be NULL) when seed is baked in. */
int32_t cudnn_fe_graph_rng(cudnnFeGraph*, const cudnnFeTensor* seed, const cudnnFeTensor* offset,
                           const cudnnFeRngAttrs*, cudnnFeTensor** y);

typedef struct cudnnFeNormAttrs {
    cudnnFeNormFwdPhase forward_phase;
    cudnnFeDataType compute_data_type;
    int32_t has_epsilon_scalar; float epsilon_scalar;
    const char* name;
} cudnnFeNormAttrs;

int32_t cudnn_fe_graph_layernorm(cudnnFeGraph*, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                                 const cudnnFeTensor* bias, const cudnnFeTensor* epsilon,
                                 const cudnnFeNormAttrs*, cudnnFeTensor** out, size_t out_len);
int32_t cudnn_fe_graph_instancenorm(cudnnFeGraph*, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                                    const cudnnFeTensor* bias, const cudnnFeTensor* epsilon,
                                    const cudnnFeNormAttrs*, cudnnFeTensor** out, size_t out_len);
int32_t cudnn_fe_graph_adalayernorm(cudnnFeGraph*, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                                    const cudnnFeTensor* bias, const cudnnFeTensor* epsilon,
                                    const cudnnFeNormAttrs*, cudnnFeTensor** out, size_t out_len);
int32_t cudnn_fe_graph_rmsnorm(cudnnFeGraph*, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                               const cudnnFeTensor* bias, const cudnnFeTensor* epsilon,
                               const cudnnFeNormAttrs*, cudnnFeTensor** out, size_t out_len);

typedef struct cudnnFeNormBwdAttrs {
    cudnnFeDataType compute_data_type;
    int32_t has_dbias; int32_t has_dbias_value;
    const char* name;
} cudnnFeNormBwdAttrs;
int32_t cudnn_fe_graph_layernorm_backward(cudnnFeGraph*, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                                          const cudnnFeTensor* scale, const cudnnFeTensor* mean,
                                          const cudnnFeTensor* inv_variance, const cudnnFeNormBwdAttrs*,
                                          cudnnFeTensor** out, size_t out_len);
int32_t cudnn_fe_graph_instancenorm_backward(cudnnFeGraph*, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                                             const cudnnFeTensor* scale, const cudnnFeTensor* mean,
                                             const cudnnFeTensor* inv_variance, const cudnnFeNormBwdAttrs*,
                                             cudnnFeTensor** out, size_t out_len);
int32_t cudnn_fe_graph_adalayernorm_backward(cudnnFeGraph*, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                                             const cudnnFeTensor* scale, const cudnnFeTensor* mean,
                                             const cudnnFeTensor* inv_variance, const cudnnFeNormBwdAttrs*,
                                             cudnnFeTensor** out, size_t out_len);
int32_t cudnn_fe_graph_rmsnorm_backward(cudnnFeGraph*, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                                        const cudnnFeTensor* scale, const cudnnFeTensor* inv_variance,
                                        const cudnnFeNormBwdAttrs*, cudnnFeTensor** out, size_t out_len);

int32_t cudnn_fe_graph_batchnorm(cudnnFeGraph*, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                                 const cudnnFeTensor* bias, const cudnnFeTensor* in_running_mean,
                                 const cudnnFeTensor* in_running_var, const cudnnFeTensor* epsilon,
                                 const cudnnFeTensor* momentum, const cudnnFeTensor* const* peer_stats,
                                 size_t n_peer_stats, const cudnnFeNormAttrs*,
                                 cudnnFeTensor** out, size_t out_len);
int32_t cudnn_fe_graph_batchnorm_inference(cudnnFeGraph*, const cudnnFeTensor* x, const cudnnFeTensor* mean,
                                           const cudnnFeTensor* inv_variance, const cudnnFeTensor* scale,
                                           const cudnnFeTensor* bias, const cudnnFeNormAttrs*,
                                           cudnnFeTensor** y);
int32_t cudnn_fe_graph_batchnorm_backward(cudnnFeGraph*, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                                          const cudnnFeTensor* scale, const cudnnFeTensor* mean,
                                          const cudnnFeTensor* inv_variance,
                                          const cudnnFeTensor* const* peer_stats, size_t n_peer_stats,
                                          const cudnnFeNormBwdAttrs*, cudnnFeTensor** out, size_t out_len);
int32_t cudnn_fe_graph_genstats(cudnnFeGraph*, const cudnnFeTensor* x, const cudnnFeNormAttrs*,
                                cudnnFeTensor** out, size_t out_len);
int32_t cudnn_fe_graph_dbn_weight(cudnnFeGraph*, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                                  const cudnnFeTensor* scale, const cudnnFeTensor* mean,
                                  const cudnnFeTensor* inv_variance, const cudnnFeNormAttrs*,
                                  cudnnFeTensor** out, size_t out_len);
int32_t cudnn_fe_graph_bn_finalize(cudnnFeGraph*, const cudnnFeTensor* sum, const cudnnFeTensor* sq_sum,
                                   const cudnnFeTensor* scale, const cudnnFeTensor* bias,
                                   const cudnnFeTensor* epsilon, const cudnnFeTensor* accum_count,
                                   const cudnnFeNormAttrs*, cudnnFeTensor** out, size_t out_len);

typedef struct cudnnFeRoPEAttrs {
    cudnnFeDataType compute_data_type;
    int32_t has_output_scale; float output_scale;
    int32_t has_rope_dim;     int64_t rope_dim;
    const char* name;
} cudnnFeRoPEAttrs;
int32_t cudnn_fe_graph_rope(cudnnFeGraph*, const cudnnFeTensor* x, const cudnnFeTensor* freqs,
                            const cudnnFeRoPEAttrs*, cudnnFeTensor** y);
int32_t cudnn_fe_graph_rope_backward(cudnnFeGraph*, const cudnnFeTensor* dy, const cudnnFeTensor* freqs,
                                     const cudnnFeRoPEAttrs*, cudnnFeTensor** dx);

typedef struct cudnnFeSdpaAttrs {
    cudnnFeDataType compute_data_type;
    int32_t generate_stats;  
    int32_t has_attn_scale;   float attn_scale;
    int32_t use_alibi_mask;
    int32_t use_padding_mask;
    int32_t use_causal_mask;
    int32_t use_causal_mask_bottom_right;
    int32_t has_diagonal_alignment;   cudnnFeDiagonalAlignment diagonal_alignment;
    int32_t has_diagonal_band_left;   int32_t diagonal_band_left;
    int32_t has_diagonal_band_right;  int32_t diagonal_band_right;
    int32_t has_sliding_window;       int32_t sliding_window_length;
    int32_t has_dropout_prob;   float dropout_probability;
    cudnnFeAttentionImplementation implementation;
    const char* name;
    /* optional input tensors (nullable) */
    const cudnnFeTensor* bias;
    const cudnnFeTensor* seq_len_q;
    const cudnnFeTensor* seq_len_kv;
    const cudnnFeTensor* dropout_seed;   /* required if has_dropout_prob */
    const cudnnFeTensor* dropout_offset; /* required if has_dropout_prob */
} cudnnFeSdpaAttrs;

int32_t cudnn_fe_graph_sdpa(cudnnFeGraph*, const cudnnFeTensor* q, const cudnnFeTensor* k,
                            const cudnnFeTensor* v, const cudnnFeSdpaAttrs*,
                            cudnnFeTensor** out, size_t out_len);

typedef struct cudnnFeSdpaBackwardAttrs {
    cudnnFeDataType compute_data_type;
    int32_t has_attn_scale;   float attn_scale;
    int32_t use_alibi_mask;
    int32_t use_padding_mask;
    int32_t use_causal_mask;
    int32_t use_causal_mask_bottom_right;
    int32_t has_diagonal_alignment;   cudnnFeDiagonalAlignment diagonal_alignment;
    int32_t has_diagonal_band_left;   int32_t diagonal_band_left;
    int32_t has_diagonal_band_right;  int32_t diagonal_band_right;
    int32_t has_sliding_window;       int32_t sliding_window_length;
    int32_t has_dropout_prob;   float dropout_probability;
    int32_t use_deterministic_algorithm;
    const char* name;
    const cudnnFeTensor* bias;
    const cudnnFeTensor* dbias;
    const cudnnFeTensor* seq_len_q;
    const cudnnFeTensor* seq_len_kv;
    const cudnnFeTensor* dropout_seed;
    const cudnnFeTensor* dropout_offset;
} cudnnFeSdpaBackwardAttrs;

int32_t cudnn_fe_graph_sdpa_backward(cudnnFeGraph*, const cudnnFeTensor* q, const cudnnFeTensor* k,
                                     const cudnnFeTensor* v, const cudnnFeTensor* o, const cudnnFeTensor* dO,
                                     const cudnnFeTensor* stats, const cudnnFeSdpaBackwardAttrs*,
                                     cudnnFeTensor** out, size_t out_len);

typedef struct cudnnFeBlockScaleQuantizeAttrs {
    cudnnFeDataType compute_data_type;
    int32_t block_size;
    int32_t has_axis; int64_t axis;
    int32_t has_transpose; int32_t transpose;
    const char* name;
} cudnnFeBlockScaleQuantizeAttrs;

int32_t cudnn_fe_graph_block_scale_quantize(cudnnFeGraph*, const cudnnFeTensor* x,
                                            const cudnnFeBlockScaleQuantizeAttrs*,
                                            cudnnFeTensor** out, size_t out_len);
typedef struct cudnnFeBlockScaleDequantizeAttrs {
    cudnnFeDataType compute_data_type;
    const int32_t* block_size; size_t block_size_len;
    int32_t has_is_negative_scale; int32_t is_negative_scale;
    const char* name;
} cudnnFeBlockScaleDequantizeAttrs;
int32_t cudnn_fe_graph_block_scale_dequantize(cudnnFeGraph*, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                                              const cudnnFeBlockScaleDequantizeAttrs*, cudnnFeTensor** y);

typedef struct cudnnFeMoeGroupedMatmulAttrs {
    cudnnFeDataType compute_data_type;
    cudnnFeMoeGroupedMatmulMode mode;
    int32_t has_top_k; int32_t top_k;
    const char* name;
} cudnnFeMoeGroupedMatmulAttrs;

int32_t cudnn_fe_graph_moe_grouped_matmul(cudnnFeGraph*, const cudnnFeTensor* token, const cudnnFeTensor* weight,
                                          const cudnnFeTensor* first_token_offset, const cudnnFeTensor* token_index,
                                          const cudnnFeTensor* token_ks, const cudnnFeMoeGroupedMatmulAttrs*,
                                          cudnnFeTensor** out);
typedef struct cudnnFeMoeGroupedMatmulBwdAttrs {
    cudnnFeDataType compute_data_type;
    const char* name;
} cudnnFeMoeGroupedMatmulBwdAttrs;
int32_t cudnn_fe_graph_moe_grouped_matmul_bwd(cudnnFeGraph*, const cudnnFeTensor* doutput, const cudnnFeTensor* token,
                                              const cudnnFeTensor* first_token_offset,
                                              const cudnnFeMoeGroupedMatmulBwdAttrs*, cudnnFeTensor** dweight);

int32_t cudnn_fe_graph_validate(cudnnFeGraph*);
int32_t cudnn_fe_graph_build_operation_graph(cudnnFeGraph*, cudnnFeHandle*);
int32_t cudnn_fe_graph_create_execution_plans(cudnnFeGraph*, const cudnnFeHeurMode* modes, size_t n_modes);
int32_t cudnn_fe_graph_check_support(cudnnFeGraph*, cudnnFeHandle*);
int32_t cudnn_fe_graph_build_plans(cudnnFeGraph*, cudnnFeHandle*, cudnnFeBuildPlanPolicy);
int32_t cudnn_fe_graph_build(cudnnFeGraph*, cudnnFeHandle*, const cudnnFeHeurMode* modes, size_t n_modes);

int32_t cudnn_fe_graph_deselect_workspace_greater_than(cudnnFeGraph*, int64_t workspace);
int32_t cudnn_fe_graph_deselect_engines(cudnnFeGraph*, const char* const* names, size_t n_names);

int32_t cudnn_fe_graph_get_execution_plan_count(cudnnFeGraph*, int64_t* out);
int32_t cudnn_fe_graph_get_engine_count(cudnnFeGraph*, int64_t* out);
int32_t cudnn_fe_graph_get_plan_name_at_index(cudnnFeGraph*, int64_t index, char* buf, size_t buf_len);

int32_t cudnn_fe_graph_get_workspace_size(cudnnFeGraph*, int64_t* out);

/* Per-tensor runtime shape override, used when the graph was built with
 * cudnn_fe_graph_set_override_shape_enabled. Tensors not listed keep their
 * build-time shape. */
typedef struct cudnnFeShapeOverride {
    int64_t        uid;
    const int64_t* dim;
    size_t         dim_len;
    const int64_t* stride;
    size_t         stride_len;
} cudnnFeShapeOverride;

/* Workspace size for the given shape overrides. Requires cuDNN backend >= 9.23.0.
 * With n_overrides == 0 this behaves like cudnn_fe_graph_get_workspace_size. */
int32_t cudnn_fe_graph_get_workspace_size_with_overrides(cudnnFeGraph*, cudnnFeHandle*,
                                                         const cudnnFeShapeOverride* overrides,
                                                         size_t n_overrides, int64_t* out);

/* variant pack given as parallel arrays of length n: uid[i] -> dev_ptr[i].
 * `overrides` may be NULL when n_overrides == 0. */
int32_t cudnn_fe_graph_execute(cudnnFeGraph*, cudnnFeHandle*,
                               const int64_t* uids, const uint64_t* dev_ptrs, size_t n,
                               uint64_t workspace,
                               const cudnnFeShapeOverride* overrides, size_t n_overrides);
int32_t cudnn_fe_graph_execute_plan_at_index(cudnnFeGraph*, cudnnFeHandle*,
                                             const int64_t* uids, const uint64_t* dev_ptrs, size_t n,
                                             uint64_t workspace, int64_t index,
                                             const cudnnFeShapeOverride* overrides, size_t n_overrides);

int32_t cudnn_fe_graph_serialize(cudnnFeGraph*, uint8_t** out_data, size_t* out_len);
void    cudnn_fe_buffer_free(uint8_t* data, size_t len);
int32_t cudnn_fe_graph_deserialize(cudnnFeGraph*, cudnnFeHandle*, const uint8_t* data, size_t len);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CUDNN_FE_SHIM_H */
