#include "cudnn_fe_shim.h"

#include "cudnn_frontend.h"

#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>

namespace fe = cudnn_frontend;

namespace cudnn_frontend {
void* cudnn_dlhandle = nullptr;
}  // namespace cudnn_frontend

using fe::graph::Graph;
using fe::graph::Tensor_attributes;
using Tensor_t = std::shared_ptr<Tensor_attributes>;

#define FE_ASSERT_ENUM(C_VAL, CPP_VAL) \
    static_assert((int)(C_VAL) == (int)(CPP_VAL), "enum drift: " #C_VAL " != " #CPP_VAL)

// error_code_t
FE_ASSERT_ENUM(CUDNN_FE_OK, fe::error_code_t::OK);
FE_ASSERT_ENUM(CUDNN_FE_INVALID_VALUE, fe::error_code_t::INVALID_VALUE);
FE_ASSERT_ENUM(CUDNN_FE_NVRTC_COMPILATION_FAILED, fe::error_code_t::NVRTC_COMPILATION_FAILED);
FE_ASSERT_ENUM(CUDNN_FE_GRAPH_NOT_SUPPORTED, fe::error_code_t::GRAPH_NOT_SUPPORTED);
// DataType_t
FE_ASSERT_ENUM(CUDNN_FE_DATA_TYPE_NOT_SET, fe::DataType_t::NOT_SET);
FE_ASSERT_ENUM(CUDNN_FE_DATA_TYPE_FLOAT, fe::DataType_t::FLOAT);
FE_ASSERT_ENUM(CUDNN_FE_DATA_TYPE_HALF, fe::DataType_t::HALF);
FE_ASSERT_ENUM(CUDNN_FE_DATA_TYPE_BFLOAT16, fe::DataType_t::BFLOAT16);
FE_ASSERT_ENUM(CUDNN_FE_DATA_TYPE_FP8_E4M3, fe::DataType_t::FP8_E4M3);
FE_ASSERT_ENUM(CUDNN_FE_DATA_TYPE_COMPLEX_FP64, fe::DataType_t::COMPLEX_FP64);
// HeurMode_t (A/B/FALLBACK share values; OPENSOURCE excluded)
FE_ASSERT_ENUM(CUDNN_FE_HEUR_MODE_A, fe::HeurMode_t::A);
FE_ASSERT_ENUM(CUDNN_FE_HEUR_MODE_B, fe::HeurMode_t::B);
FE_ASSERT_ENUM(CUDNN_FE_HEUR_MODE_FALLBACK, fe::HeurMode_t::FALLBACK);
// BuildPlanPolicy_t
FE_ASSERT_ENUM(CUDNN_FE_BUILD_PLAN_POLICY_HEURISTICS_CHOICE, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
FE_ASSERT_ENUM(CUDNN_FE_BUILD_PLAN_POLICY_ALL, fe::BuildPlanPolicy_t::ALL);
// PointwiseMode_t
FE_ASSERT_ENUM(CUDNN_FE_POINTWISE_NOT_SET, fe::PointwiseMode_t::NOT_SET);
FE_ASSERT_ENUM(CUDNN_FE_POINTWISE_ADD, fe::PointwiseMode_t::ADD);
FE_ASSERT_ENUM(CUDNN_FE_POINTWISE_BINARY_SELECT, fe::PointwiseMode_t::BINARY_SELECT);
FE_ASSERT_ENUM(CUDNN_FE_POINTWISE_RECIPROCAL, fe::PointwiseMode_t::RECIPROCAL);
// ReductionMode_t
FE_ASSERT_ENUM(CUDNN_FE_REDUCTION_NOT_SET, fe::ReductionMode_t::NOT_SET);
FE_ASSERT_ENUM(CUDNN_FE_REDUCTION_ADD, fe::ReductionMode_t::ADD);
FE_ASSERT_ENUM(CUDNN_FE_REDUCTION_MUL_NO_ZEROS, fe::ReductionMode_t::MUL_NO_ZEROS);
// NormFwdPhase_t
FE_ASSERT_ENUM(CUDNN_FE_NORM_FWD_NOT_SET, fe::NormFwdPhase_t::NOT_SET);
FE_ASSERT_ENUM(CUDNN_FE_NORM_FWD_TRAINING, fe::NormFwdPhase_t::TRAINING);
// ConvolutionMode_t
FE_ASSERT_ENUM(CUDNN_FE_CONV_MODE_NOT_SET, fe::ConvolutionMode_t::NOT_SET);
FE_ASSERT_ENUM(CUDNN_FE_CONV_MODE_CROSS_CORRELATION, fe::ConvolutionMode_t::CROSS_CORRELATION);
// ResampleMode_t
FE_ASSERT_ENUM(CUDNN_FE_RESAMPLE_NOT_SET, fe::ResampleMode_t::NOT_SET);
FE_ASSERT_ENUM(CUDNN_FE_RESAMPLE_MAXPOOL, fe::ResampleMode_t::MAXPOOL);
// PaddingMode_t
FE_ASSERT_ENUM(CUDNN_FE_PADDING_NOT_SET, fe::PaddingMode_t::NOT_SET);
FE_ASSERT_ENUM(CUDNN_FE_PADDING_ZERO_PAD, fe::PaddingMode_t::ZERO_PAD);
// ReshapeMode_t
FE_ASSERT_ENUM(CUDNN_FE_RESHAPE_NOT_SET, fe::ReshapeMode_t::NOT_SET);
FE_ASSERT_ENUM(CUDNN_FE_RESHAPE_LOGICAL, fe::ReshapeMode_t::LOGICAL);
// TensorReordering_t
FE_ASSERT_ENUM(CUDNN_FE_TENSOR_REORDERING_NONE, fe::TensorReordering_t::NONE);
FE_ASSERT_ENUM(CUDNN_FE_TENSOR_REORDERING_F8_128x4, fe::TensorReordering_t::F8_128x4);
// DiagonalAlignment_t
FE_ASSERT_ENUM(CUDNN_FE_DIAGONAL_ALIGNMENT_TOP_LEFT, fe::DiagonalAlignment_t::TOP_LEFT);
FE_ASSERT_ENUM(CUDNN_FE_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT, fe::DiagonalAlignment_t::BOTTOM_RIGHT);
// RngDistribution_t
FE_ASSERT_ENUM(CUDNN_FE_RNG_DIST_NOT_SET, fe::RngDistribution_t::NOT_SET);
FE_ASSERT_ENUM(CUDNN_FE_RNG_DIST_NORMAL, fe::RngDistribution_t::NORMAL);
// MoeGroupedMatmulMode_t
FE_ASSERT_ENUM(CUDNN_FE_MOE_GMM_NOT_SET, fe::MoeGroupedMatmulMode_t::NOT_SET);
FE_ASSERT_ENUM(CUDNN_FE_MOE_GMM_SCATTER, fe::MoeGroupedMatmulMode_t::SCATTER);
// AttentionImplementation_t
FE_ASSERT_ENUM(CUDNN_FE_ATTENTION_IMPL_AUTO, fe::AttentionImplementation_t::AUTO);
FE_ASSERT_ENUM(CUDNN_FE_ATTENTION_IMPL_UNIFIED, fe::AttentionImplementation_t::UNIFIED);

struct cudnnFeGraph {
    std::shared_ptr<Graph> g = std::make_shared<Graph>();
    // Keeps every Tensor_attributes handed out alive for the graph's lifetime,
    // and lets us recover the shared_ptr from a raw handle.
    std::unordered_map<Tensor_attributes*, Tensor_t> registry;

    Tensor_t keep(Tensor_t t) {
        if (t) registry[t.get()] = t;
        return t;
    }
};

struct cudnnFeHandle {
    cudnnHandle_t h = nullptr;
    bool owns       = false;
};

struct cudnnFeKernelCache {
    std::shared_ptr<fe::KernelCache> kc;
};

static thread_local std::string g_last_error;

extern "C" const char*
cudnn_fe_last_error_message(void) {
    return g_last_error.c_str();
}

// Error conversion helper.
static int32_t
run(const std::function<fe::error_t()>& f) {
    try {
        fe::error_t e = f();
        if (e.is_bad()) {
            g_last_error = e.get_message();
            return static_cast<int32_t>(e.get_code());
        }
        g_last_error.clear();
        return CUDNN_FE_OK;
    } catch (const std::exception& ex) {
        g_last_error = ex.what();
        return CUDNN_FE_SHIM_EXCEPTION;
    } catch (...) {
        g_last_error = "unknown C++ exception in cudnn-frontend shim";
        return CUDNN_FE_SHIM_EXCEPTION;
    }
}

static const fe::error_t OK = {fe::error_code_t::OK, ""};

#define FE_CHECK(x)                       \
    do {                                  \
        fe::error_t _e = (x);             \
        if (_e.is_bad()) return _e;       \
    } while (0)

static inline fe::DataType_t
DT(cudnnFeDataType v) {
    return static_cast<fe::DataType_t>(v);
}

static inline Tensor_t
as_t(cudnnFeGraph* g, const cudnnFeTensor* h) {
    if (!h) return nullptr;
    auto it = g->registry.find(reinterpret_cast<Tensor_attributes*>(const_cast<cudnnFeTensor*>(h)));
    return it == g->registry.end() ? nullptr : it->second;
}

static inline cudnnFeTensor*
as_h(const Tensor_t& t) {
    return reinterpret_cast<cudnnFeTensor*>(t.get());
}

static inline std::vector<int64_t>
vec(const int64_t* p, size_t n) {
    return (p && n) ? std::vector<int64_t>(p, p + n) : std::vector<int64_t>{};
}

// returns 1 if loaded. Not thread-safe.
static int
ensure_cudnn_loaded(const char* explicit_path) {
    if (fe::cudnn_dlhandle) return 1;
    std::vector<const char*> candidates;
    if (explicit_path && explicit_path[0]) candidates.push_back(explicit_path);
    candidates.push_back("libcudnn.so");
    candidates.push_back("libcudnn.so.9");
    for (const char* name : candidates) {
        void* h = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
        if (h) {
            fe::cudnn_dlhandle = h;
            return 1;
        }
    }
    return 0;
}

extern "C" int32_t
cudnn_fe_load_library(const char* path) {
    return run([&]() -> fe::error_t {
        if (!ensure_cudnn_loaded(path ? path : "")) {
            return {fe::error_code_t::HANDLE_ERROR,
                    std::string("failed to dlopen libcudnn: ") + (dlerror() ? dlerror() : "unknown")};
        }
        return OK;
    });
}

extern "C" int64_t
cudnn_fe_backend_version(void) {
    ensure_cudnn_loaded(nullptr);
    return static_cast<int64_t>(fe::detail::get_backend_version());
}

extern "C" int32_t
cudnn_fe_handle_create(void* cuda_stream, cudnnFeHandle** out) {
    return run([&]() -> fe::error_t {
        if (!ensure_cudnn_loaded(nullptr)) {
            return {fe::error_code_t::HANDLE_ERROR, "failed to dlopen libcudnn"};
        }
        auto* wrap = new cudnnFeHandle();
        cudnnStatus_t s = fe::detail::create_handle(&wrap->h);
        if (s != CUDNN_STATUS_SUCCESS) {
            delete wrap;
            return {fe::error_code_t::HANDLE_ERROR, "cudnnCreate failed"};
        }
        wrap->owns = true;
        if (cuda_stream) {
            fe::detail::set_stream(wrap->h, static_cast<cudaStream_t>(cuda_stream));
        }
        *out = wrap;
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_handle_set_stream(cudnnFeHandle* h, void* cuda_stream) {
    return run([&]() -> fe::error_t {
        fe::detail::set_stream(h->h, static_cast<cudaStream_t>(cuda_stream));
        return OK;
    });
}

extern "C" void
cudnn_fe_handle_destroy(cudnnFeHandle* h) {
    if (!h) return;
    if (h->owns && h->h) fe::detail::destroy_handle(h->h);
    delete h;
}

extern "C" int32_t
cudnn_fe_kernel_cache_create(cudnnFeKernelCache** out) {
    return run([&]() -> fe::error_t {
        auto* wrap = new cudnnFeKernelCache();
        wrap->kc   = std::make_shared<fe::KernelCache>();
        *out       = wrap;
        return OK;
    });
}

extern "C" void
cudnn_fe_kernel_cache_destroy(cudnnFeKernelCache* c) {
    delete c;
}

extern "C" int32_t
cudnn_fe_graph_create(cudnnFeGraph** out) {
    return run([&]() -> fe::error_t {
        *out = new cudnnFeGraph();
        return OK;
    });
}

extern "C" void
cudnn_fe_graph_destroy(cudnnFeGraph* g) {
    delete g;
}

#define GRAPH_SET(fn, cppcall)                            \
    extern "C" int32_t fn(cudnnFeGraph* g, cudnnFeDataType v) { \
        return run([&]() -> fe::error_t {                 \
            g->g->cppcall(DT(v));                         \
            return OK;                                    \
        });                                               \
    }
GRAPH_SET(cudnn_fe_graph_set_io_data_type, set_io_data_type)
GRAPH_SET(cudnn_fe_graph_set_intermediate_data_type, set_intermediate_data_type)
GRAPH_SET(cudnn_fe_graph_set_compute_data_type, set_compute_data_type)
#undef GRAPH_SET

extern "C" int32_t
cudnn_fe_graph_set_name(cudnnFeGraph* g, const char* name) {
    return run([&]() -> fe::error_t {
        g->g->set_name(name ? name : "");
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_set_dynamic_shape_enabled(cudnnFeGraph* g, int32_t enabled) {
    return run([&]() -> fe::error_t {
        g->g->set_dynamic_shape_enabled(enabled != 0);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_set_override_shape_enabled(cudnnFeGraph* g, int32_t enabled) {
    return run([&]() -> fe::error_t {
        g->g->set_override_shape_enabled(enabled != 0);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_set_sm_count(cudnnFeGraph* g, int32_t sm_count) {
    return run([&]() -> fe::error_t {
        g->g->set_sm_count(sm_count);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_set_sm_version(cudnnFeGraph* g, int32_t sm_version) {
    return run([&]() -> fe::error_t {
        g->g->set_sm_version(sm_version);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_set_kernel_cache(cudnnFeGraph* g, cudnnFeKernelCache* c) {
    return run([&]() -> fe::error_t {
        g->g->set_kernel_cache(c ? c->kc : nullptr);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_tensor(cudnnFeGraph* g, const cudnnFeTensorDesc* d, cudnnFeTensor** out) {
    return run([&]() -> fe::error_t {
        Tensor_attributes ta;
        ta.set_dim(vec(d->dim, d->dim_len)).set_stride(vec(d->stride, d->stride_len));
        if (d->data_type != CUDNN_FE_DATA_TYPE_NOT_SET) ta.set_data_type(DT(d->data_type));
        ta.set_is_virtual(d->is_virtual != 0);
        ta.set_is_pass_by_value(d->is_pass_by_value != 0);
        if (d->has_uid) ta.set_uid(d->uid);
        if (d->alignment) ta.set_alignment(d->alignment);
        ta.set_reordering_type(static_cast<fe::TensorReordering_t>(d->reordering));
        if (d->ragged_offset) ta.set_ragged_offset(as_t(g, d->ragged_offset));
        if (d->ragged_offset_multiplier) ta.set_ragged_offset_multiplier(d->ragged_offset_multiplier);
        if (d->name) ta.set_name(d->name);
        Tensor_t t = g->keep(g->g->tensor(ta));
        *out       = as_h(t);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_tensor_like(cudnnFeGraph* g, const cudnnFeTensor* like, const char* name, cudnnFeTensor** out) {
    return run([&]() -> fe::error_t {
        Tensor_t t = g->keep(g->g->tensor_like(as_t(g, like), name ? name : std::string{}));
        *out       = as_h(t);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_get_uid(const cudnnFeTensor* t, int64_t* out) {
    return run([&]() -> fe::error_t {
        *out = reinterpret_cast<const Tensor_attributes*>(t)->get_uid();
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_set_uid(cudnnFeTensor* t, int64_t uid) {
    return run([&]() -> fe::error_t {
        reinterpret_cast<Tensor_attributes*>(t)->set_uid(uid);
        return OK;
    });
}

static int32_t
copy_dims(const std::vector<int64_t>& src, int64_t* out, size_t* len) {
    size_t n = src.size();
    if (out) {
        size_t m = (*len < n) ? *len : n;
        for (size_t i = 0; i < m; ++i) out[i] = src[i];
    }
    *len = n;
    return CUDNN_FE_OK;
}

extern "C" int32_t
cudnn_fe_tensor_get_dim(const cudnnFeTensor* t, int64_t* out, size_t* len) {
    return run([&]() -> fe::error_t {
        copy_dims(reinterpret_cast<const Tensor_attributes*>(t)->get_dim(), out, len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_get_stride(const cudnnFeTensor* t, int64_t* out, size_t* len) {
    return run([&]() -> fe::error_t {
        copy_dims(reinterpret_cast<const Tensor_attributes*>(t)->get_stride(), out, len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_set_output(cudnnFeTensor* t, int32_t is_output) {
    return run([&]() -> fe::error_t {
        reinterpret_cast<Tensor_attributes*>(t)->set_output(is_output != 0);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_get_data_type(const cudnnFeTensor* t, cudnnFeDataType* out) {
    return run([&]() -> fe::error_t {
        *out = static_cast<cudnnFeDataType>(reinterpret_cast<const Tensor_attributes*>(t)->get_data_type());
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_get_is_virtual(const cudnnFeTensor* t, int32_t* out) {
    return run([&]() -> fe::error_t {
        *out = reinterpret_cast<const Tensor_attributes*>(t)->get_is_virtual() ? 1 : 0;
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_get_is_pass_by_value(const cudnnFeTensor* t, int32_t* out) {
    return run([&]() -> fe::error_t {
        *out = reinterpret_cast<const Tensor_attributes*>(t)->get_is_pass_by_value() ? 1 : 0;
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_get_reordering(const cudnnFeTensor* t, cudnnFeTensorReordering* out) {
    return run([&]() -> fe::error_t {
        *out = static_cast<cudnnFeTensorReordering>(
            reinterpret_cast<const Tensor_attributes*>(t)->get_reordering_type());
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_data_type_size_bits(cudnnFeDataType dt, int64_t* out) {
    return run([&]() -> fe::error_t {
        *out = static_cast<int64_t>(fe::detail::get_element_size_in_bits(DT(dt)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_set_data_type(cudnnFeTensor* t, cudnnFeDataType dt) {
    return run([&]() -> fe::error_t {
        reinterpret_cast<Tensor_attributes*>(t)->set_data_type(DT(dt));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_set_ragged_offset(cudnnFeGraph* g, cudnnFeTensor* t, const cudnnFeTensor* offset, int64_t multiplier) {
    return run([&]() -> fe::error_t {
        Tensor_t off = as_t(g, offset);
        if (!off) return {fe::error_code_t::INVALID_VALUE, "ragged offset tensor does not belong to this graph"};
        auto* ta = reinterpret_cast<Tensor_attributes*>(t);
        ta->set_ragged_offset(off);
        if (multiplier) ta->set_ragged_offset_multiplier(multiplier);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_get_has_ragged_offset(const cudnnFeTensor* t, int32_t* out) {
    return run([&]() -> fe::error_t {
        *out = reinterpret_cast<Tensor_attributes*>(const_cast<cudnnFeTensor*>(t))->get_ragged_offset() ? 1 : 0;
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_set_dim(cudnnFeTensor* t, const int64_t* dim, size_t len) {
    return run([&]() -> fe::error_t {
        reinterpret_cast<Tensor_attributes*>(t)->set_dim(vec(dim, len));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_tensor_set_stride(cudnnFeTensor* t, const int64_t* stride, size_t len) {
    return run([&]() -> fe::error_t {
        reinterpret_cast<Tensor_attributes*>(t)->set_stride(vec(stride, len));
        return OK;
    });
}

template <typename Arr>
static void
emit_outputs(cudnnFeGraph* g, const Arr& arr, cudnnFeTensor** out, size_t out_len) {
    size_t i = 0;
    for (const auto& t : arr) {
        if (i >= out_len) break;
        out[i++] = t ? as_h(g->keep(t)) : nullptr;
    }
    for (; i < out_len; ++i) out[i] = nullptr;
}

extern "C" int32_t
cudnn_fe_graph_matmul(cudnnFeGraph* g, const cudnnFeTensor* a, const cudnnFeTensor* b,
                      const cudnnFeMatmulAttrs* attrs, cudnnFeTensor** c) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Matmul_attributes().set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->has_padding) at.set_padding(attrs->padding);
        if (attrs->name) at.set_name(attrs->name);
        *c = as_h(g->keep(g->g->matmul(as_t(g, a), as_t(g, b), at)));
        return OK;
    });
}

static fe::graph::Pointwise_attributes
make_pointwise(const cudnnFePointwiseAttrs* a) {
    auto at = fe::graph::Pointwise_attributes()
                  .set_mode(static_cast<fe::PointwiseMode_t>(a->mode))
                  .set_compute_data_type(DT(a->compute_data_type));
    if (a->has_axis) at.set_axis(a->axis);
    if (a->has_relu_lower_clip_slope) at.set_relu_lower_clip_slope(a->relu_lower_clip_slope);
    if (a->has_relu_lower_clip) at.set_relu_lower_clip(a->relu_lower_clip);
    if (a->has_relu_upper_clip) at.set_relu_upper_clip(a->relu_upper_clip);
    if (a->has_swish_beta) at.set_swish_beta(a->swish_beta);
    if (a->has_elu_alpha) at.set_elu_alpha(a->elu_alpha);
    if (a->has_softplus_beta) at.set_softplus_beta(a->softplus_beta);
    if (a->name) at.set_name(a->name);
    return at;
}

extern "C" int32_t
cudnn_fe_graph_pointwise_unary(cudnnFeGraph* g, const cudnnFeTensor* a, const cudnnFePointwiseAttrs* attrs,
                               cudnnFeTensor** out) {
    return run([&]() -> fe::error_t {
        *out = as_h(g->keep(g->g->pointwise(as_t(g, a), make_pointwise(attrs))));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_pointwise_binary(cudnnFeGraph* g, const cudnnFeTensor* a, const cudnnFeTensor* b,
                                const cudnnFePointwiseAttrs* attrs, cudnnFeTensor** out) {
    return run([&]() -> fe::error_t {
        *out = as_h(g->keep(g->g->pointwise(as_t(g, a), as_t(g, b), make_pointwise(attrs))));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_pointwise_ternary(cudnnFeGraph* g, const cudnnFeTensor* a, const cudnnFeTensor* b,
                                 const cudnnFeTensor* c, const cudnnFePointwiseAttrs* attrs, cudnnFeTensor** out) {
    return run([&]() -> fe::error_t {
        *out = as_h(g->keep(g->g->pointwise(as_t(g, a), as_t(g, b), as_t(g, c), make_pointwise(attrs))));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_reduction(cudnnFeGraph* g, const cudnnFeTensor* a, const cudnnFeReductionAttrs* attrs,
                         cudnnFeTensor** out) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Reduction_attributes()
                      .set_mode(static_cast<fe::ReductionMode_t>(attrs->mode))
                      .set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->has_is_deterministic) at.set_is_deterministic(attrs->is_deterministic != 0);
        if (attrs->name) at.set_name(attrs->name);
        *out = as_h(g->keep(g->g->reduction(as_t(g, a), at)));
        return OK;
    });
}

template <typename Attr>
static Attr
make_conv(const cudnnFeConvAttrs* a) {
    Attr at;
    at.set_compute_data_type(DT(a->compute_data_type));
    if (a->conv_mode != CUDNN_FE_CONV_MODE_NOT_SET)
        at.set_convolution_mode(static_cast<fe::ConvolutionMode_t>(a->conv_mode));
    if (a->pre_padding_len) at.set_pre_padding(vec(a->pre_padding, a->pre_padding_len));
    if (a->post_padding_len) at.set_post_padding(vec(a->post_padding, a->post_padding_len));
    if (a->stride_len) at.set_stride(vec(a->stride, a->stride_len));
    if (a->dilation_len) at.set_dilation(vec(a->dilation, a->dilation_len));
    if (a->name) at.set_name(a->name);
    return at;
}

extern "C" int32_t
cudnn_fe_graph_conv_fprop(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeTensor* w,
                          const cudnnFeConvAttrs* attrs, cudnnFeTensor** y) {
    return run([&]() -> fe::error_t {
        *y = as_h(g->keep(g->g->conv_fprop(as_t(g, x), as_t(g, w), make_conv<fe::graph::Conv_fprop_attributes>(attrs))));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_conv_dgrad(cudnnFeGraph* g, const cudnnFeTensor* dy, const cudnnFeTensor* w,
                          const cudnnFeConvAttrs* attrs, cudnnFeTensor** dx) {
    return run([&]() -> fe::error_t {
        *dx = as_h(g->keep(g->g->conv_dgrad(as_t(g, dy), as_t(g, w), make_conv<fe::graph::Conv_dgrad_attributes>(attrs))));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_conv_wgrad(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeTensor* dy,
                          const cudnnFeConvAttrs* attrs, cudnnFeTensor** dw) {
    return run([&]() -> fe::error_t {
        *dw = as_h(g->keep(g->g->conv_wgrad(as_t(g, x), as_t(g, dy), make_conv<fe::graph::Conv_wgrad_attributes>(attrs))));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_reshape(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeReshapeAttrs* attrs, cudnnFeTensor** y) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Reshape_attributes();
        if (attrs->mode != CUDNN_FE_RESHAPE_NOT_SET) at.set_reshape_mode(static_cast<fe::ReshapeMode_t>(attrs->mode));
        if (attrs->dim_len) at.set_dim(vec(attrs->dim, attrs->dim_len));
        if (attrs->stride_len) at.set_stride(vec(attrs->stride, attrs->stride_len));
        if (attrs->name) at.set_name(attrs->name);
        *y = as_h(g->keep(g->g->reshape(as_t(g, x), at)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_transpose(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeTransposeAttrs* attrs,
                         cudnnFeTensor** y) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Transpose_attributes().set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->permutation_len) at.set_permutation(vec(attrs->permutation, attrs->permutation_len));
        if (attrs->name) at.set_name(attrs->name);
        *y = as_h(g->keep(g->g->transpose(as_t(g, x), at)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_slice(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeSliceAttrs* attrs, cudnnFeTensor** y) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Slice_attributes().set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->slices_len) {
            std::vector<std::pair<int64_t, int64_t>> slices;
            for (size_t i = 0; i + 1 < attrs->slices_len; i += 2)
                slices.emplace_back(attrs->slices[i], attrs->slices[i + 1]);
            at.set_slices(slices);
        }
        if (attrs->name) at.set_name(attrs->name);
        *y = as_h(g->keep(g->g->slice(as_t(g, x), at)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_concatenate(cudnnFeGraph* g, const cudnnFeTensor* const* inputs, size_t n_inputs,
                           const cudnnFeConcatenateAttrs* attrs, cudnnFeTensor** y) {
    return run([&]() -> fe::error_t {
        std::vector<Tensor_t> ins;
        ins.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) ins.push_back(as_t(g, inputs[i]));
        auto at = fe::graph::Concatenate_attributes()
                      .set_compute_data_type(DT(attrs->compute_data_type))
                      .set_axis(attrs->axis);
        if (attrs->has_in_place_index) at.set_in_place_index(attrs->in_place_index);
        if (attrs->name) at.set_name(attrs->name);
        *y = as_h(g->keep(g->g->concatenate(ins, at)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_resample(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeResampleAttrs* attrs,
                        cudnnFeTensor** y, cudnnFeTensor** index) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Resample_attributes()
                      .set_compute_data_type(DT(attrs->compute_data_type))
                      .set_resampling_mode(static_cast<fe::ResampleMode_t>(attrs->resample_mode));
        if (attrs->padding_mode != CUDNN_FE_PADDING_NOT_SET)
            at.set_padding_mode(static_cast<fe::PaddingMode_t>(attrs->padding_mode));
        if (attrs->has_is_inference) at.set_is_inference(attrs->is_inference != 0);
        at.set_generate_index(attrs->generate_index != 0);
        if (attrs->window_len) at.set_window(vec(attrs->window, attrs->window_len));
        if (attrs->stride_len) at.set_stride(vec(attrs->stride, attrs->stride_len));
        if (attrs->pre_padding_len) at.set_pre_padding(vec(attrs->pre_padding, attrs->pre_padding_len));
        if (attrs->post_padding_len) at.set_post_padding(vec(attrs->post_padding, attrs->post_padding_len));
        if (attrs->name) at.set_name(attrs->name);
        auto res = g->g->resample(as_t(g, x), at); 
        *y       = as_h(g->keep(res[0]));
        if (index) *index = res[1] ? as_h(g->keep(res[1])) : nullptr;
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_rng(cudnnFeGraph* g, const cudnnFeTensor* seed, const cudnnFeTensor* offset,
                   const cudnnFeRngAttrs* attrs, cudnnFeTensor** y) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Rng_attributes()
                      .set_compute_data_type(DT(attrs->compute_data_type))
                      .set_distribution(static_cast<fe::RngDistribution_t>(attrs->distribution));
        if (attrs->has_seed) at.set_seed(attrs->seed);
        if (attrs->has_prob) at.set_bernoulli_probability(attrs->bernoulli_probability);
        if (attrs->dim_len) at.set_dim(vec(attrs->dim, attrs->dim_len));
        if (attrs->stride_len) at.set_stride(vec(attrs->stride, attrs->stride_len));
        if (attrs->name) at.set_name(attrs->name);
        *y = as_h(g->keep(g->g->rng(as_t(g, seed), as_t(g, offset), at)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_layernorm(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                         const cudnnFeTensor* bias, const cudnnFeTensor* epsilon, const cudnnFeNormAttrs* attrs,
                         cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto eps = as_t(g, epsilon);
        auto at  = fe::graph::Layernorm_attributes()
                      .set_forward_phase(static_cast<fe::NormFwdPhase_t>(attrs->forward_phase))
                      .set_compute_data_type(DT(attrs->compute_data_type))
                      .set_epsilon(eps);
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g, g->g->layernorm(as_t(g, x), as_t(g, scale), as_t(g, bias), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_instancenorm(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                            const cudnnFeTensor* bias, const cudnnFeTensor* epsilon, const cudnnFeNormAttrs* attrs,
                            cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto eps = as_t(g, epsilon);
        auto at  = fe::graph::Instancenorm_attributes()
                      .set_forward_phase(static_cast<fe::NormFwdPhase_t>(attrs->forward_phase))
                      .set_compute_data_type(DT(attrs->compute_data_type))
                      .set_epsilon(eps);
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g, g->g->instancenorm(as_t(g, x), as_t(g, scale), as_t(g, bias), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_adalayernorm(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                            const cudnnFeTensor* bias, const cudnnFeTensor* epsilon, const cudnnFeNormAttrs* attrs,
                            cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto eps = as_t(g, epsilon);
        auto at  = fe::graph::AdaLayernorm_attributes()
                      .set_forward_phase(static_cast<fe::NormFwdPhase_t>(attrs->forward_phase))
                      .set_compute_data_type(DT(attrs->compute_data_type))
                      .set_epsilon(eps);
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g, g->g->adalayernorm(as_t(g, x), as_t(g, scale), as_t(g, bias), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_rmsnorm(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                       const cudnnFeTensor* bias, const cudnnFeTensor* epsilon, const cudnnFeNormAttrs* attrs,
                       cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto eps  = as_t(g, epsilon);
        auto bias_t = as_t(g, bias);
        auto at   = fe::graph::Rmsnorm_attributes()
                      .set_forward_phase(static_cast<fe::NormFwdPhase_t>(attrs->forward_phase))
                      .set_compute_data_type(DT(attrs->compute_data_type));
        if (bias_t) at.set_bias(bias_t);
        if (eps) at.set_epsilon(eps);
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g, g->g->rmsnorm(as_t(g, x), as_t(g, scale), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_layernorm_backward(cudnnFeGraph* g, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                                  const cudnnFeTensor* scale, const cudnnFeTensor* mean,
                                  const cudnnFeTensor* inv_variance, const cudnnFeNormBwdAttrs* attrs,
                                  cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Layernorm_backward_attributes()
                      .set_saved_mean_and_inv_variance(as_t(g, mean), as_t(g, inv_variance))
                      .set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g, g->g->layernorm_backward(as_t(g, dy), as_t(g, x), as_t(g, scale), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_instancenorm_backward(cudnnFeGraph* g, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                                     const cudnnFeTensor* scale, const cudnnFeTensor* mean,
                                     const cudnnFeTensor* inv_variance, const cudnnFeNormBwdAttrs* attrs,
                                     cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Instancenorm_backward_attributes()
                      .set_saved_mean_and_inv_variance(as_t(g, mean), as_t(g, inv_variance))
                      .set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g, g->g->instancenorm_backward(as_t(g, dy), as_t(g, x), as_t(g, scale), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_adalayernorm_backward(cudnnFeGraph* g, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                                     const cudnnFeTensor* scale, const cudnnFeTensor* mean,
                                     const cudnnFeTensor* inv_variance, const cudnnFeNormBwdAttrs* attrs,
                                     cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::AdaLayernorm_backward_attributes()
                      .set_saved_mean_and_inv_variance(as_t(g, mean), as_t(g, inv_variance))
                      .set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g, g->g->adalayernorm_backward(as_t(g, dy), as_t(g, x), as_t(g, scale), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_rmsnorm_backward(cudnnFeGraph* g, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                                const cudnnFeTensor* scale, const cudnnFeTensor* inv_variance,
                                const cudnnFeNormBwdAttrs* attrs, cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Rmsnorm_backward_attributes()
                      .has_dbias(attrs->has_dbias ? (attrs->has_dbias_value != 0) : true)
                      .set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g, g->g->rmsnorm_backward(as_t(g, dy), as_t(g, x), as_t(g, scale), as_t(g, inv_variance), at), out,
                     out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_batchnorm(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                         const cudnnFeTensor* bias, const cudnnFeTensor* in_running_mean,
                         const cudnnFeTensor* in_running_var, const cudnnFeTensor* epsilon,
                         const cudnnFeTensor* momentum, const cudnnFeTensor* const* peer_stats, size_t n_peer_stats,
                         const cudnnFeNormAttrs* attrs, cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto in_mean = as_t(g, in_running_mean);
        auto in_var  = as_t(g, in_running_var);
        auto mom     = as_t(g, momentum);
        auto eps     = as_t(g, epsilon);
        std::vector<Tensor_t> peers;
        for (size_t i = 0; i < n_peer_stats; ++i) peers.push_back(as_t(g, peer_stats[i]));
        auto at = fe::graph::Batchnorm_attributes()
                      .set_compute_data_type(DT(attrs->compute_data_type))
                      .set_epsilon(eps)
                      .set_previous_running_stats(in_mean, in_var, mom)
                      .set_peer_stats(peers);
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g, g->g->batchnorm(as_t(g, x), as_t(g, scale), as_t(g, bias), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_batchnorm_inference(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeTensor* mean,
                                   const cudnnFeTensor* inv_variance, const cudnnFeTensor* scale,
                                   const cudnnFeTensor* bias, const cudnnFeNormAttrs* attrs, cudnnFeTensor** y) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Batchnorm_inference_attributes().set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->name) at.set_name(attrs->name);
        *y = as_h(g->keep(g->g->batchnorm_inference(as_t(g, x), as_t(g, mean), as_t(g, inv_variance), as_t(g, scale),
                                                    as_t(g, bias), at)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_batchnorm_backward(cudnnFeGraph* g, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                                  const cudnnFeTensor* scale, const cudnnFeTensor* mean,
                                  const cudnnFeTensor* inv_variance, const cudnnFeTensor* const* peer_stats,
                                  size_t n_peer_stats, const cudnnFeNormBwdAttrs* attrs, cudnnFeTensor** out,
                                  size_t out_len) {
    return run([&]() -> fe::error_t {
        std::vector<Tensor_t> peers;
        for (size_t i = 0; i < n_peer_stats; ++i) peers.push_back(as_t(g, peer_stats[i]));
        auto at = fe::graph::Batchnorm_backward_attributes()
                      .set_saved_mean_and_inv_variance(as_t(g, mean), as_t(g, inv_variance))
                      .set_peer_stats(peers)
                      .set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g, g->g->batchnorm_backward(as_t(g, dy), as_t(g, x), as_t(g, scale), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_genstats(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeNormAttrs* attrs, cudnnFeTensor** out,
                        size_t out_len) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Genstats_attributes().set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g, g->g->genstats(as_t(g, x), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_dbn_weight(cudnnFeGraph* g, const cudnnFeTensor* dy, const cudnnFeTensor* x,
                          const cudnnFeTensor* scale, const cudnnFeTensor* mean, const cudnnFeTensor* inv_variance,
                          const cudnnFeNormAttrs* attrs, cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::DBN_weight_attributes().set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g,
                     g->g->dbn_weight(as_t(g, dy), as_t(g, x), as_t(g, mean), as_t(g, inv_variance), as_t(g, scale), at),
                     out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_bn_finalize(cudnnFeGraph* g, const cudnnFeTensor* sum, const cudnnFeTensor* sq_sum,
                           const cudnnFeTensor* scale, const cudnnFeTensor* bias, const cudnnFeTensor* epsilon,
                           const cudnnFeTensor* accum_count, const cudnnFeNormAttrs* attrs, cudnnFeTensor** out,
                           size_t out_len) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::BN_finalize_attributes().set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->name) at.set_name(attrs->name);
        emit_outputs(g,
                     g->g->bn_finalize(as_t(g, sum), as_t(g, sq_sum), as_t(g, scale), as_t(g, bias), as_t(g, epsilon),
                                       as_t(g, accum_count), at),
                     out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_rope(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeTensor* freqs, const cudnnFeRoPEAttrs* attrs,
                    cudnnFeTensor** y) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::RoPE_attributes().set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->has_output_scale) at.set_output_scale(attrs->output_scale);
        if (attrs->has_rope_dim) at.set_rope_dim(attrs->rope_dim);
        if (attrs->name) at.set_name(attrs->name);
        *y = as_h(g->keep(g->g->rope(as_t(g, x), as_t(g, freqs), at)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_rope_backward(cudnnFeGraph* g, const cudnnFeTensor* dy, const cudnnFeTensor* freqs,
                             const cudnnFeRoPEAttrs* attrs, cudnnFeTensor** dx) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::RoPE_backward_attributes().set_compute_data_type(DT(attrs->compute_data_type));
        if (attrs->has_output_scale) at.set_output_scale(attrs->output_scale);
        if (attrs->has_rope_dim) at.set_rope_dim(attrs->rope_dim);
        if (attrs->name) at.set_name(attrs->name);
        *dx = as_h(g->keep(g->g->rope_backward(as_t(g, dy), as_t(g, freqs), at)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_sdpa(cudnnFeGraph* g, const cudnnFeTensor* q, const cudnnFeTensor* k, const cudnnFeTensor* v,
                    const cudnnFeSdpaAttrs* a, cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::SDPA_attributes()
                      .set_compute_data_type(DT(a->compute_data_type))
                      .set_generate_stats(a->generate_stats != 0);
        if (a->has_attn_scale) at.set_attn_scale(a->attn_scale);
        if (a->bias) at.set_bias(as_t(g, a->bias));
        at.set_alibi_mask(a->use_alibi_mask != 0);
        at.set_padding_mask(a->use_padding_mask != 0);
        if (a->seq_len_q) at.set_seq_len_q(as_t(g, a->seq_len_q));
        if (a->seq_len_kv) at.set_seq_len_kv(as_t(g, a->seq_len_kv));
        if (a->use_causal_mask) at.set_causal_mask(true);
        if (a->use_causal_mask_bottom_right) at.set_causal_mask_bottom_right(true);
        if (a->has_diagonal_alignment) at.set_diagonal_alignment(static_cast<fe::DiagonalAlignment_t>(a->diagonal_alignment));
        if (a->has_diagonal_band_left) at.set_diagonal_band_left_bound(a->diagonal_band_left);
        if (a->has_diagonal_band_right) at.set_diagonal_band_right_bound(a->diagonal_band_right);
        if (a->has_sliding_window) at.set_sliding_window_length(a->sliding_window_length);
        if (a->has_dropout_prob)
            at.set_dropout(a->dropout_probability, as_t(g, a->dropout_seed), as_t(g, a->dropout_offset));
        at.set_implementation(static_cast<fe::AttentionImplementation_t>(a->implementation));
        if (a->name) at.set_name(a->name);
        emit_outputs(g, g->g->sdpa(as_t(g, q), as_t(g, k), as_t(g, v), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_sdpa_backward(cudnnFeGraph* g, const cudnnFeTensor* q, const cudnnFeTensor* k, const cudnnFeTensor* v,
                             const cudnnFeTensor* o, const cudnnFeTensor* dO, const cudnnFeTensor* stats,
                             const cudnnFeSdpaBackwardAttrs* a, cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::SDPA_backward_attributes().set_compute_data_type(DT(a->compute_data_type));
        if (a->has_attn_scale) at.set_attn_scale(a->attn_scale);
        if (a->bias) at.set_bias(as_t(g, a->bias));
        if (a->dbias) at.set_dbias(as_t(g, a->dbias));
        at.set_alibi_mask(a->use_alibi_mask != 0);
        at.set_padding_mask(a->use_padding_mask != 0);
        if (a->seq_len_q) at.set_seq_len_q(as_t(g, a->seq_len_q));
        if (a->seq_len_kv) at.set_seq_len_kv(as_t(g, a->seq_len_kv));
        if (a->use_causal_mask) at.set_causal_mask(true);
        if (a->use_causal_mask_bottom_right) at.set_causal_mask_bottom_right(true);
        if (a->has_diagonal_alignment) at.set_diagonal_alignment(static_cast<fe::DiagonalAlignment_t>(a->diagonal_alignment));
        if (a->has_diagonal_band_left) at.set_diagonal_band_left_bound(a->diagonal_band_left);
        if (a->has_diagonal_band_right) at.set_diagonal_band_right_bound(a->diagonal_band_right);
        if (a->has_sliding_window) at.set_sliding_window_length(a->sliding_window_length);
        if (a->has_dropout_prob)
            at.set_dropout(a->dropout_probability, as_t(g, a->dropout_seed), as_t(g, a->dropout_offset));
        if (a->use_deterministic_algorithm) at.set_deterministic_algorithm(true);
        if (a->name) at.set_name(a->name);
        emit_outputs(g, g->g->sdpa_backward(as_t(g, q), as_t(g, k), as_t(g, v), as_t(g, o), as_t(g, dO), as_t(g, stats), at),
                     out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_block_scale_quantize(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeBlockScaleQuantizeAttrs* a,
                                    cudnnFeTensor** out, size_t out_len) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Block_scale_quantize_attributes()
                      .set_compute_data_type(DT(a->compute_data_type))
                      .set_block_size(a->block_size);
        if (a->has_axis) at.set_axis(a->axis);
        if (a->has_transpose) at.set_transpose(a->transpose != 0);
        if (a->name) at.set_name(a->name);
        emit_outputs(g, g->g->block_scale_quantize(as_t(g, x), at), out, out_len);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_block_scale_dequantize(cudnnFeGraph* g, const cudnnFeTensor* x, const cudnnFeTensor* scale,
                                      const cudnnFeBlockScaleDequantizeAttrs* a, cudnnFeTensor** y) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Block_scale_dequantize_attributes().set_compute_data_type(DT(a->compute_data_type));
        if (a->block_size_len)
            at.set_block_size(std::vector<int32_t>(a->block_size, a->block_size + a->block_size_len));
        if (a->has_is_negative_scale) at.set_is_negative_scale(a->is_negative_scale != 0);
        if (a->name) at.set_name(a->name);
        *y = as_h(g->keep(g->g->block_scale_dequantize(as_t(g, x), as_t(g, scale), at)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_moe_grouped_matmul(cudnnFeGraph* g, const cudnnFeTensor* token, const cudnnFeTensor* weight,
                                  const cudnnFeTensor* first_token_offset, const cudnnFeTensor* token_index,
                                  const cudnnFeTensor* token_ks, const cudnnFeMoeGroupedMatmulAttrs* a,
                                  cudnnFeTensor** out) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Moe_grouped_matmul_attributes()
                      .set_compute_data_type(DT(a->compute_data_type))
                      .set_mode(static_cast<fe::MoeGroupedMatmulMode_t>(a->mode));
        if (a->has_top_k) at.set_top_k(a->top_k);
        if (a->name) at.set_name(a->name);
        *out = as_h(g->keep(g->g->moe_grouped_matmul(as_t(g, token), as_t(g, weight), as_t(g, first_token_offset),
                                                     as_t(g, token_index), as_t(g, token_ks), at)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_moe_grouped_matmul_bwd(cudnnFeGraph* g, const cudnnFeTensor* doutput, const cudnnFeTensor* token,
                                      const cudnnFeTensor* first_token_offset,
                                      const cudnnFeMoeGroupedMatmulBwdAttrs* a, cudnnFeTensor** dweight) {
    return run([&]() -> fe::error_t {
        auto at = fe::graph::Moe_grouped_matmul_bwd_attributes().set_compute_data_type(DT(a->compute_data_type));
        if (a->name) at.set_name(a->name);
        *dweight = as_h(g->keep(g->g->moe_grouped_matmul_bwd(as_t(g, doutput), as_t(g, token),
                                                             as_t(g, first_token_offset), at)));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_validate(cudnnFeGraph* g) {
    return run([&]() -> fe::error_t { return g->g->validate(); });
}

extern "C" int32_t
cudnn_fe_graph_build_operation_graph(cudnnFeGraph* g, cudnnFeHandle* h) {
    return run([&]() -> fe::error_t { return g->g->build_operation_graph(h->h); });
}

extern "C" int32_t
cudnn_fe_graph_create_execution_plans(cudnnFeGraph* g, const cudnnFeHeurMode* modes, size_t n_modes) {
    return run([&]() -> fe::error_t {
        std::vector<fe::HeurMode_t> ms;
        for (size_t i = 0; i < n_modes; ++i) ms.push_back(static_cast<fe::HeurMode_t>(modes[i]));
        return g->g->create_execution_plans(ms);
    });
}

extern "C" int32_t
cudnn_fe_graph_check_support(cudnnFeGraph* g, cudnnFeHandle* h) {
    return run([&]() -> fe::error_t { return g->g->check_support(h->h); });
}

extern "C" int32_t
cudnn_fe_graph_build_plans(cudnnFeGraph* g, cudnnFeHandle* h, cudnnFeBuildPlanPolicy policy) {
    return run([&]() -> fe::error_t {
        return g->g->build_plans(h->h, static_cast<fe::BuildPlanPolicy_t>(policy));
    });
}

extern "C" int32_t
cudnn_fe_graph_build(cudnnFeGraph* g, cudnnFeHandle* h, const cudnnFeHeurMode* modes, size_t n_modes) {
    return run([&]() -> fe::error_t {
        std::vector<fe::HeurMode_t> ms;
        for (size_t i = 0; i < n_modes; ++i) ms.push_back(static_cast<fe::HeurMode_t>(modes[i]));
        FE_CHECK(g->g->validate());
        FE_CHECK(g->g->build_operation_graph(h->h));
        FE_CHECK(g->g->create_execution_plans(ms));
        FE_CHECK(g->g->check_support(h->h));
        FE_CHECK(g->g->build_plans(h->h, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_deselect_workspace_greater_than(cudnnFeGraph* g, int64_t workspace) {
    return run([&]() -> fe::error_t {
        g->g->deselect_workspace_greater_than(workspace);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_deselect_engines(cudnnFeGraph* g, const char* const* names, size_t n_names) {
    return run([&]() -> fe::error_t {
        std::vector<std::string> ns;
        for (size_t i = 0; i < n_names; ++i) ns.emplace_back(names[i]);
        g->g->deselect_engines(ns);
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_get_execution_plan_count(cudnnFeGraph* g, int64_t* out) {
    return run([&]() -> fe::error_t {
        *out = g->g->get_execution_plan_count();
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_get_engine_count(cudnnFeGraph* g, int64_t* out) {
    return run([&]() -> fe::error_t {
        int64_t count = 0;
        FE_CHECK(g->g->get_engine_count(count));
        *out = count;
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_get_plan_name_at_index(cudnnFeGraph* g, int64_t index, char* buf, size_t buf_len) {
    return run([&]() -> fe::error_t {
        std::string name;
        FE_CHECK(g->g->get_plan_name_at_index(index, name));
        if (buf && buf_len) {
            std::strncpy(buf, name.c_str(), buf_len - 1);
            buf[buf_len - 1] = '\0';
        }
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_get_workspace_size(cudnnFeGraph* g, int64_t* out) {
    return run([&]() -> fe::error_t {
        int64_t sz = 0;
        FE_CHECK(g->g->get_workspace_size(sz));
        *out = sz;
        return OK;
    });
}

static std::unordered_map<int64_t, void*>
build_varpack(const int64_t* uids, const uint64_t* dev_ptrs, size_t n) {
    std::unordered_map<int64_t, void*> m;
    m.reserve(n);
    for (size_t i = 0; i < n; ++i)
        m.emplace(uids[i], reinterpret_cast<void*>(static_cast<uintptr_t>(dev_ptrs[i])));
    return m;
}

// Splits the flat C array into the three parallel vectors the frontend expects.
static void
unpack_overrides(const cudnnFeShapeOverride* overrides, size_t n, std::vector<int64_t>& uids,
                 std::vector<std::vector<int64_t>>& shapes, std::vector<std::vector<int64_t>>& strides) {
    uids.reserve(n);
    shapes.reserve(n);
    strides.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        uids.push_back(overrides[i].uid);
        shapes.push_back(vec(overrides[i].dim, overrides[i].dim_len));
        strides.push_back(vec(overrides[i].stride, overrides[i].stride_len));
    }
}

extern "C" int32_t
cudnn_fe_graph_get_workspace_size_with_overrides(cudnnFeGraph* g, cudnnFeHandle* h,
                                                 const cudnnFeShapeOverride* overrides, size_t n_overrides,
                                                 int64_t* out) {
    return run([&]() -> fe::error_t {
        int64_t sz = 0;
        if (n_overrides == 0) {
            FE_CHECK(g->g->get_workspace_size(sz));
        } else {
            std::vector<int64_t> uids;
            std::vector<std::vector<int64_t>> shapes, strides;
            unpack_overrides(overrides, n_overrides, uids, shapes, strides);
            FE_CHECK(g->g->get_workspace_size(h->h, sz, uids, shapes, strides));
        }
        *out = sz;
        return OK;
    });
}

extern "C" int32_t
cudnn_fe_graph_execute(cudnnFeGraph* g, cudnnFeHandle* h, const int64_t* uids, const uint64_t* dev_ptrs, size_t n,
                       uint64_t workspace, const cudnnFeShapeOverride* overrides, size_t n_overrides) {
    return run([&]() -> fe::error_t {
        auto vp    = build_varpack(uids, dev_ptrs, n);
        auto* ws_p = reinterpret_cast<void*>(static_cast<uintptr_t>(workspace));
        if (n_overrides == 0) return g->g->execute(h->h, vp, ws_p);
        std::vector<int64_t> o_uids;
        std::vector<std::vector<int64_t>> o_shapes, o_strides;
        unpack_overrides(overrides, n_overrides, o_uids, o_shapes, o_strides);
        return g->g->execute(h->h, vp, ws_p, o_uids, o_shapes, o_strides);
    });
}

extern "C" int32_t
cudnn_fe_graph_execute_plan_at_index(cudnnFeGraph* g, cudnnFeHandle* h, const int64_t* uids, const uint64_t* dev_ptrs,
                                     size_t n, uint64_t workspace, int64_t index,
                                     const cudnnFeShapeOverride* overrides, size_t n_overrides) {
    return run([&]() -> fe::error_t {
        auto vp    = build_varpack(uids, dev_ptrs, n);
        auto* ws_p = reinterpret_cast<void*>(static_cast<uintptr_t>(workspace));
        if (n_overrides == 0) return g->g->execute_plan_at_index(h->h, vp, ws_p, index);
        std::vector<int64_t> o_uids;
        std::vector<std::vector<int64_t>> o_shapes, o_strides;
        unpack_overrides(overrides, n_overrides, o_uids, o_shapes, o_strides);
        return g->g->execute_plan_at_index(h->h, vp, ws_p, index, o_uids, o_shapes, o_strides);
    });
}

extern "C" int32_t
cudnn_fe_graph_serialize(cudnnFeGraph* g, uint8_t** out_data, size_t* out_len) {
    return run([&]() -> fe::error_t {
        std::vector<uint8_t> data;
        FE_CHECK(g->g->serialize(data));
        auto* buf = new uint8_t[data.size()];
        std::memcpy(buf, data.data(), data.size());
        *out_data = buf;
        *out_len  = data.size();
        return OK;
    });
}

extern "C" void
cudnn_fe_buffer_free(uint8_t* data, size_t) {
    delete[] data;
}

extern "C" int32_t
cudnn_fe_graph_deserialize(cudnnFeGraph* g, cudnnFeHandle* h, const uint8_t* data, size_t len) {
    return run([&]() -> fe::error_t {
        std::vector<uint8_t> v(data, data + len);
        return g->g->deserialize(h->h, v);
    });
}
