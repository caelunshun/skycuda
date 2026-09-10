use std::ptr;

use cudnn_frontend_sys as sys;

use crate::error::check;
use crate::{
    AttentionImplementation, ConvolutionMode, DiagonalAlignment, MoeGroupedMatmulMode,
    NormFwdPhase, PaddingMode, PointwiseMode, ReductionMode, ResampleMode, ReshapeMode,
    RngDistribution,
};
use crate::{DataType, GraphBuilder, Result, Tensor, cstr, opt_ptr};

unsafe fn collect_outputs<'g, 'h, const N: usize>(
    raw: [*mut sys::cudnnFeTensor; N],
    graph: &'g GraphBuilder<'h>,
) -> [Option<Tensor<'g, 'h>>; N] {
    let mut out: [Option<Tensor<'g, 'h>>; N] = [None; N];
    for i in 0..N {
        out[i] = if raw[i].is_null() {
            None
        } else {
            Some(unsafe { Tensor::from_raw(raw[i], graph) })
        };
    }
    out
}

/// Attributes for [`Graph::matmul`].
#[derive(Clone, Default)]
pub struct MatmulAttributes {
    pub compute_data_type: Option<DataType>,
    pub padding: Option<f64>,
    pub name: Option<String>,
}

impl MatmulAttributes {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn padding(mut self, p: f64) -> Self {
        self.padding = Some(p);
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

impl<'h> GraphBuilder<'h> {
    /// `C = A @ B` (batched matmul).
    pub fn matmul<'g>(
        &'g self,
        a: Tensor<'g, 'h>,
        b: Tensor<'g, 'h>,
        attrs: &MatmulAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(a.is_in_graph(self));
        assert!(b.is_in_graph(self));

        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeMatmulAttrs {
            compute_data_type: attrs.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            has_padding: attrs.padding.is_some() as i32,
            padding: attrs.padding.unwrap_or(0.0),
            name,
        };
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_matmul(self.raw, a.as_ptr(), b.as_ptr(), &raw, &mut out)
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
}

#[derive(Clone)]
pub struct PointwiseAttributes {
    pub mode: PointwiseMode,
    pub compute_data_type: Option<DataType>,
    pub axis: Option<i64>,
    pub relu_lower_clip_slope: Option<f32>,
    pub relu_lower_clip: Option<f32>,
    pub relu_upper_clip: Option<f32>,
    pub swish_beta: Option<f32>,
    pub elu_alpha: Option<f32>,
    pub softplus_beta: Option<f32>,
    pub name: Option<String>,
}

impl PointwiseAttributes {
    pub fn new(mode: PointwiseMode) -> Self {
        PointwiseAttributes {
            mode,
            compute_data_type: None,
            axis: None,
            relu_lower_clip_slope: None,
            relu_lower_clip: None,
            relu_upper_clip: None,
            swish_beta: None,
            elu_alpha: None,
            softplus_beta: None,
            name: None,
        }
    }

    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }

    pub fn axis(mut self, a: i64) -> Self {
        self.axis = Some(a);
        self
    }

    pub fn relu_lower_clip_slope(mut self, v: f32) -> Self {
        self.relu_lower_clip_slope = Some(v);
        self
    }

    pub fn relu_lower_clip(mut self, v: f32) -> Self {
        self.relu_lower_clip = Some(v);
        self
    }

    pub fn relu_upper_clip(mut self, v: f32) -> Self {
        self.relu_upper_clip = Some(v);
        self
    }

    pub fn swish_beta(mut self, v: f32) -> Self {
        self.swish_beta = Some(v);
        self
    }

    pub fn elu_alpha(mut self, v: f32) -> Self {
        self.elu_alpha = Some(v);
        self
    }

    pub fn softplus_beta(mut self, v: f32) -> Self {
        self.softplus_beta = Some(v);
        self
    }

    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }

    fn to_raw(&self, name: *const std::os::raw::c_char) -> sys::cudnnFePointwiseAttrs {
        sys::cudnnFePointwiseAttrs {
            mode: self.mode.to_ffi(),
            compute_data_type: self.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            has_axis: self.axis.is_some() as i32,
            axis: self.axis.unwrap_or(0),
            has_relu_lower_clip_slope: self.relu_lower_clip_slope.is_some() as i32,
            relu_lower_clip_slope: self.relu_lower_clip_slope.unwrap_or(0.0),
            has_relu_lower_clip: self.relu_lower_clip.is_some() as i32,
            relu_lower_clip: self.relu_lower_clip.unwrap_or(0.0),
            has_relu_upper_clip: self.relu_upper_clip.is_some() as i32,
            relu_upper_clip: self.relu_upper_clip.unwrap_or(0.0),
            has_swish_beta: self.swish_beta.is_some() as i32,
            swish_beta: self.swish_beta.unwrap_or(0.0),
            has_elu_alpha: self.elu_alpha.is_some() as i32,
            elu_alpha: self.elu_alpha.unwrap_or(0.0),
            has_softplus_beta: self.softplus_beta.is_some() as i32,
            softplus_beta: self.softplus_beta.unwrap_or(0.0),
            name,
        }
    }
}

impl<'h> GraphBuilder<'h> {
    pub fn pointwise_unary<'g>(
        &'g self,
        a: Tensor,
        attrs: &PointwiseAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(a.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_pointwise_unary(self.raw, a.as_ptr(), &raw, &mut out)
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }

    pub fn pointwise_binary<'g>(
        &'g self,
        a: Tensor,
        b: Tensor,
        attrs: &PointwiseAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(a.is_in_graph(self));
        assert!(b.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_pointwise_binary(self.raw, a.as_ptr(), b.as_ptr(), &raw, &mut out)
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }

    pub fn pointwise_ternary<'g>(
        &'g self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        attrs: &PointwiseAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(a.is_in_graph(self));
        assert!(b.is_in_graph(self));
        assert!(c.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_pointwise_ternary(
                self.raw,
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                &raw,
                &mut out,
            )
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
}

#[derive(Clone)]
pub struct ReductionAttributes {
    pub mode: ReductionMode,
    pub compute_data_type: Option<DataType>,
    pub is_deterministic: Option<bool>,
    pub name: Option<String>,
}

impl ReductionAttributes {
    pub fn new(mode: ReductionMode) -> Self {
        ReductionAttributes {
            mode,
            compute_data_type: None,
            is_deterministic: None,
            name: None,
        }
    }

    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }

    pub fn is_deterministic(mut self, v: bool) -> Self {
        self.is_deterministic = Some(v);
        self
    }

    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

impl<'h> GraphBuilder<'h> {
    pub fn reduction<'g>(
        &'g self,
        a: Tensor,
        attrs: &ReductionAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(a.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeReductionAttrs {
            mode: attrs.mode.to_ffi(),
            compute_data_type: attrs.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            has_is_deterministic: attrs.is_deterministic.is_some() as i32,
            is_deterministic: attrs.is_deterministic.unwrap_or(false) as i32,
            name,
        };
        let mut out = ptr::null_mut();
        check(unsafe { sys::cudnn_fe_graph_reduction(self.raw, a.as_ptr(), &raw, &mut out) })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
}

#[derive(Clone, Default)]
pub struct ConvAttributes {
    pub compute_data_type: Option<DataType>,
    pub conv_mode: Option<ConvolutionMode>,
    pub pre_padding: Vec<i64>,
    pub post_padding: Vec<i64>,
    pub stride: Vec<i64>,
    pub dilation: Vec<i64>,
    pub name: Option<String>,
}

impl ConvAttributes {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn conv_mode(mut self, m: ConvolutionMode) -> Self {
        self.conv_mode = Some(m);
        self
    }
    pub fn pre_padding(mut self, v: &[i64]) -> Self {
        self.pre_padding = v.to_vec();
        self
    }
    pub fn post_padding(mut self, v: &[i64]) -> Self {
        self.post_padding = v.to_vec();
        self
    }
    pub fn padding(mut self, v: &[i64]) -> Self {
        self.pre_padding = v.to_vec();
        self.post_padding = v.to_vec();
        self
    }
    pub fn stride(mut self, v: &[i64]) -> Self {
        self.stride = v.to_vec();
        self
    }
    pub fn dilation(mut self, v: &[i64]) -> Self {
        self.dilation = v.to_vec();
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }

    fn to_raw(&self, name: *const std::os::raw::c_char) -> sys::cudnnFeConvAttrs {
        sys::cudnnFeConvAttrs {
            compute_data_type: self.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            conv_mode: self.conv_mode.unwrap_or(ConvolutionMode::NotSet).to_ffi(),
            pre_padding: self.pre_padding.as_ptr(),
            pre_padding_len: self.pre_padding.len(),
            post_padding: self.post_padding.as_ptr(),
            post_padding_len: self.post_padding.len(),
            stride: self.stride.as_ptr(),
            stride_len: self.stride.len(),
            dilation: self.dilation.as_ptr(),
            dilation_len: self.dilation.len(),
            name,
        }
    }
}

impl<'h> GraphBuilder<'h> {
    pub fn conv_fprop<'g>(
        &'g self,
        x: Tensor,
        w: Tensor,
        attrs: &ConvAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(x.is_in_graph(self));
        assert!(w.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_conv_fprop(self.raw, x.as_ptr(), w.as_ptr(), &raw, &mut out)
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
    pub fn conv_dgrad<'g>(
        &'g self,
        dy: Tensor,
        w: Tensor,
        attrs: &ConvAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(dy.is_in_graph(self));
        assert!(w.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_conv_dgrad(self.raw, dy.as_ptr(), w.as_ptr(), &raw, &mut out)
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
    pub fn conv_wgrad<'g>(
        &'g self,
        x: Tensor,
        dy: Tensor,
        attrs: &ConvAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(x.is_in_graph(self));
        assert!(dy.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_conv_wgrad(self.raw, x.as_ptr(), dy.as_ptr(), &raw, &mut out)
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
}

#[derive(Clone, Default)]
pub struct ReshapeAttributes {
    pub mode: Option<ReshapeMode>,
    pub dim: Vec<i64>,
    pub stride: Vec<i64>,
    pub name: Option<String>,
}
impl ReshapeAttributes {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn mode(mut self, m: ReshapeMode) -> Self {
        self.mode = Some(m);
        self
    }
    pub fn dim(mut self, v: &[i64]) -> Self {
        self.dim = v.to_vec();
        self
    }
    pub fn stride(mut self, v: &[i64]) -> Self {
        self.stride = v.to_vec();
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

#[derive(Clone, Default)]
pub struct TransposeAttributes {
    pub compute_data_type: Option<DataType>,
    pub permutation: Vec<i64>,
    pub name: Option<String>,
}
impl TransposeAttributes {
    pub fn new(permutation: &[i64]) -> Self {
        TransposeAttributes {
            compute_data_type: None,
            permutation: permutation.to_vec(),
            name: None,
        }
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

#[derive(Clone, Default)]
pub struct SliceAttributes {
    pub compute_data_type: Option<DataType>,
    /// Flat `[start0, end0, start1, end1, ...]`.
    pub slices: Vec<i64>,
    pub name: Option<String>,
}
impl SliceAttributes {
    /// `ranges[i] = (start, end)` for dimension `i`.
    pub fn new(ranges: &[(i64, i64)]) -> Self {
        let mut slices = Vec::with_capacity(ranges.len() * 2);
        for (s, e) in ranges {
            slices.push(*s);
            slices.push(*e);
        }
        SliceAttributes {
            compute_data_type: None,
            slices,
            name: None,
        }
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

impl<'h> GraphBuilder<'h> {
    pub fn reshape<'g>(&'g self, x: Tensor, attrs: &ReshapeAttributes) -> Result<Tensor<'g, 'h>> {
        assert!(x.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeReshapeAttrs {
            mode: attrs.mode.unwrap_or(ReshapeMode::NotSet).to_ffi(),
            dim: attrs.dim.as_ptr(),
            dim_len: attrs.dim.len(),
            stride: attrs.stride.as_ptr(),
            stride_len: attrs.stride.len(),
            name,
        };
        let mut out = ptr::null_mut();
        check(unsafe { sys::cudnn_fe_graph_reshape(self.raw, x.as_ptr(), &raw, &mut out) })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
    pub fn transpose<'g>(
        &'g self,
        x: Tensor,
        attrs: &TransposeAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(x.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeTransposeAttrs {
            compute_data_type: attrs.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            permutation: attrs.permutation.as_ptr(),
            permutation_len: attrs.permutation.len(),
            name,
        };
        let mut out = ptr::null_mut();
        check(unsafe { sys::cudnn_fe_graph_transpose(self.raw, x.as_ptr(), &raw, &mut out) })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
    pub fn slice<'g>(&'g self, x: Tensor, attrs: &SliceAttributes) -> Result<Tensor<'g, 'h>> {
        assert!(x.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeSliceAttrs {
            compute_data_type: attrs.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            slices: attrs.slices.as_ptr(),
            slices_len: attrs.slices.len(),
            name,
        };
        let mut out = ptr::null_mut();
        check(unsafe { sys::cudnn_fe_graph_slice(self.raw, x.as_ptr(), &raw, &mut out) })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
    pub fn concatenate<'g>(
        &'g self,
        inputs: &[Tensor<'g, 'h>],
        axis: i64,
        in_place_index: Option<i64>,
        compute_data_type: Option<DataType>,
        name: Option<&str>,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(inputs.iter().all(|t| t.is_in_graph(self)));
        let (_n, name_p) = cstr(name);
        let ins: Vec<*const sys::cudnnFeTensor> = inputs.iter().map(|t| t.as_ptr()).collect();
        let raw = sys::cudnnFeConcatenateAttrs {
            compute_data_type: compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            axis,
            has_in_place_index: in_place_index.is_some() as i32,
            in_place_index: in_place_index.unwrap_or(0),
            name: name_p,
        };
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_concatenate(self.raw, ins.as_ptr(), ins.len(), &raw, &mut out)
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
}

#[derive(Clone, Default)]
pub struct ResampleAttributes {
    pub compute_data_type: Option<DataType>,
    pub resample_mode: Option<ResampleMode>,
    pub padding_mode: Option<PaddingMode>,
    pub is_inference: Option<bool>,
    pub generate_index: bool,
    pub window: Vec<i64>,
    pub stride: Vec<i64>,
    pub pre_padding: Vec<i64>,
    pub post_padding: Vec<i64>,
    pub name: Option<String>,
}
impl ResampleAttributes {
    pub fn new(mode: ResampleMode) -> Self {
        ResampleAttributes {
            resample_mode: Some(mode),
            ..Default::default()
        }
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn padding_mode(mut self, m: PaddingMode) -> Self {
        self.padding_mode = Some(m);
        self
    }
    pub fn is_inference(mut self, v: bool) -> Self {
        self.is_inference = Some(v);
        self
    }
    pub fn generate_index(mut self, v: bool) -> Self {
        self.generate_index = v;
        self
    }
    pub fn window(mut self, v: &[i64]) -> Self {
        self.window = v.to_vec();
        self
    }
    pub fn stride(mut self, v: &[i64]) -> Self {
        self.stride = v.to_vec();
        self
    }
    pub fn pre_padding(mut self, v: &[i64]) -> Self {
        self.pre_padding = v.to_vec();
        self
    }
    pub fn post_padding(mut self, v: &[i64]) -> Self {
        self.post_padding = v.to_vec();
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

impl<'h> GraphBuilder<'h> {
    /// Returns `(y, index)` where `index` is `Some` only when `generate_index`.
    pub fn resample<'g>(
        &'g self,
        x: Tensor,
        attrs: &ResampleAttributes,
    ) -> Result<(Tensor<'g, 'h>, Option<Tensor<'g, 'h>>)> {
        assert!(x.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeResampleAttrs {
            compute_data_type: attrs.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            resample_mode: attrs.resample_mode.unwrap_or(ResampleMode::NotSet).to_ffi(),
            padding_mode: attrs.padding_mode.unwrap_or(PaddingMode::NotSet).to_ffi(),
            has_is_inference: attrs.is_inference.is_some() as i32,
            is_inference: attrs.is_inference.unwrap_or(false) as i32,
            generate_index: attrs.generate_index as i32,
            window: attrs.window.as_ptr(),
            window_len: attrs.window.len(),
            stride: attrs.stride.as_ptr(),
            stride_len: attrs.stride.len(),
            pre_padding: attrs.pre_padding.as_ptr(),
            pre_padding_len: attrs.pre_padding.len(),
            post_padding: attrs.post_padding.as_ptr(),
            post_padding_len: attrs.post_padding.len(),
            name,
        };
        let mut y = ptr::null_mut();
        let mut idx = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_resample(self.raw, x.as_ptr(), &raw, &mut y, &mut idx)
        })?;
        let index = if idx.is_null() {
            None
        } else {
            Some(unsafe { Tensor::from_raw(idx, self) })
        };
        Ok((unsafe { Tensor::from_raw(y, self) }, index))
    }
}

#[derive(Clone, Default)]
pub struct RngAttributes {
    pub compute_data_type: Option<DataType>,
    pub distribution: Option<RngDistribution>,
    pub seed: Option<i64>,
    pub bernoulli_probability: Option<f64>,
    pub dim: Vec<i64>,
    pub stride: Vec<i64>,
    pub name: Option<String>,
}
impl RngAttributes {
    pub fn new(distribution: RngDistribution) -> Self {
        RngAttributes {
            distribution: Some(distribution),
            ..Default::default()
        }
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn seed(mut self, s: i64) -> Self {
        self.seed = Some(s);
        self
    }
    pub fn bernoulli_probability(mut self, p: f64) -> Self {
        self.bernoulli_probability = Some(p);
        self
    }
    pub fn dim(mut self, v: &[i64]) -> Self {
        self.dim = v.to_vec();
        self
    }
    pub fn stride(mut self, v: &[i64]) -> Self {
        self.stride = v.to_vec();
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

impl<'h> GraphBuilder<'h> {
    /// `seed`/`offset` are optional
    pub fn rng<'g>(
        &'g self,
        seed: Option<Tensor<'g, 'h>>,
        offset: Option<Tensor<'g, 'h>>,
        attrs: &RngAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        if let Some(seed) = &seed {
            assert!(seed.is_in_graph(self));
        }
        if let Some(offset) = &offset {
            assert!(offset.is_in_graph(self));
        }
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeRngAttrs {
            compute_data_type: attrs.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            distribution: attrs
                .distribution
                .unwrap_or(RngDistribution::NotSet)
                .to_ffi(),
            has_seed: attrs.seed.is_some() as i32,
            seed: attrs.seed.unwrap_or(0),
            has_prob: attrs.bernoulli_probability.is_some() as i32,
            bernoulli_probability: attrs.bernoulli_probability.unwrap_or(0.0),
            dim: attrs.dim.as_ptr(),
            dim_len: attrs.dim.len(),
            stride: attrs.stride.as_ptr(),
            stride_len: attrs.stride.len(),
            name,
        };
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_rng(self.raw, opt_ptr(seed), opt_ptr(offset), &raw, &mut out)
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
}

#[derive(Clone, Default)]
pub struct NormAttributes {
    pub forward_phase: Option<NormFwdPhase>,
    pub compute_data_type: Option<DataType>,
    pub name: Option<String>,
}
impl NormAttributes {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn forward_phase(mut self, p: NormFwdPhase) -> Self {
        self.forward_phase = Some(p);
        self
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
    fn to_raw(&self, name: *const std::os::raw::c_char) -> sys::cudnnFeNormAttrs {
        sys::cudnnFeNormAttrs {
            forward_phase: self.forward_phase.unwrap_or(NormFwdPhase::NotSet).to_ffi(),
            compute_data_type: self.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            has_epsilon_scalar: 0,
            epsilon_scalar: 0.0,
            name,
        }
    }
}

#[derive(Clone, Default)]
pub struct NormBackwardAttributes {
    pub compute_data_type: Option<DataType>,
    /// For rmsnorm backward: whether to also produce `dbias`.
    pub has_dbias: Option<bool>,
    pub name: Option<String>,
}
impl NormBackwardAttributes {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn has_dbias(mut self, v: bool) -> Self {
        self.has_dbias = Some(v);
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
    fn to_raw(&self, name: *const std::os::raw::c_char) -> sys::cudnnFeNormBwdAttrs {
        sys::cudnnFeNormBwdAttrs {
            compute_data_type: self.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            has_dbias: self.has_dbias.is_some() as i32,
            has_dbias_value: self.has_dbias.unwrap_or(false) as i32,
            name,
        }
    }
}

macro_rules! fwd_norm {
    ($method:ident, $ffi:path, $doc:literal) => {
        #[doc = $doc]
        pub fn $method<'g>(
            &'g self,
            x: Tensor<'g, 'h>,
            scale: Tensor<'g, 'h>,
            bias: Option<Tensor<'g, 'h>>,
            epsilon: Option<Tensor<'g, 'h>>,
            attrs: &NormAttributes,
        ) -> Result<[Option<Tensor<'g, 'h>>; 3]> {
            assert!(x.is_in_graph(self));
            assert!(scale.is_in_graph(self));
            if let Some(bias) = &bias {
                assert!(bias.is_in_graph(self));
            }
            if let Some(epsilon) = &epsilon {
                assert!(epsilon.is_in_graph(self));
            }
            let (_n, name) = cstr(attrs.name.as_deref());
            let raw = attrs.to_raw(name);
            let mut out = [ptr::null_mut(); 3];
            check(unsafe {
                $ffi(
                    self.raw,
                    x.as_ptr(),
                    scale.as_ptr(),
                    opt_ptr(bias),
                    opt_ptr(epsilon),
                    &raw,
                    out.as_mut_ptr(),
                    out.len(),
                )
            })?;
            Ok(unsafe { collect_outputs(out, self) })
        }
    };
}

macro_rules! bwd_norm {
    ($method:ident, $ffi:path, $doc:literal) => {
        #[doc = $doc]
        pub fn $method<'g>(
            &'g self,
            dy: Tensor<'g, 'h>,
            x: Tensor<'g, 'h>,
            scale: Tensor<'g, 'h>,
            mean: Option<Tensor<'g, 'h>>,
            inv_variance: Option<Tensor<'g, 'h>>,
            attrs: &NormBackwardAttributes,
        ) -> Result<[Option<Tensor<'g, 'h>>; 3]> {
            assert!(dy.is_in_graph(self));
            assert!(x.is_in_graph(self));
            assert!(scale.is_in_graph(self));
            if let Some(mean) = &mean {
                assert!(mean.is_in_graph(self));
            }
            if let Some(inv_variance) = &inv_variance {
                assert!(inv_variance.is_in_graph(self));
            }
            let (_n, name) = cstr(attrs.name.as_deref());
            let raw = attrs.to_raw(name);
            let mut out = [ptr::null_mut(); 3];
            check(unsafe {
                $ffi(
                    self.raw,
                    dy.as_ptr(),
                    x.as_ptr(),
                    scale.as_ptr(),
                    opt_ptr(mean),
                    opt_ptr(inv_variance),
                    &raw,
                    out.as_mut_ptr(),
                    out.len(),
                )
            })?;
            Ok(unsafe { collect_outputs(out, self) })
        }
    };
}

impl<'h> GraphBuilder<'h> {
    fwd_norm!(
        layernorm,
        sys::cudnn_fe_graph_layernorm,
        "Layernorm forward: returns `[y, mean, inv_variance]`."
    );
    fwd_norm!(
        instancenorm,
        sys::cudnn_fe_graph_instancenorm,
        "Instancenorm forward: `[y, mean, inv_variance]`."
    );
    fwd_norm!(
        adalayernorm,
        sys::cudnn_fe_graph_adalayernorm,
        "AdaLayernorm forward: `[y, mean, inv_variance]`."
    );

    bwd_norm!(
        layernorm_backward,
        sys::cudnn_fe_graph_layernorm_backward,
        "Layernorm backward: `[dx, dscale, dbias]`."
    );
    bwd_norm!(
        instancenorm_backward,
        sys::cudnn_fe_graph_instancenorm_backward,
        "Instancenorm backward: `[dx, dscale, dbias]`."
    );
    bwd_norm!(
        adalayernorm_backward,
        sys::cudnn_fe_graph_adalayernorm_backward,
        "AdaLayernorm backward: `[dx, dscale, dbias]`."
    );

    pub fn rmsnorm<'g>(
        &'g self,
        x: Tensor<'g, 'h>,
        scale: Tensor<'g, 'h>,
        bias: Option<Tensor<'g, 'h>>,
        epsilon: Option<Tensor<'g, 'h>>,
        attrs: &NormAttributes,
    ) -> Result<[Option<Tensor<'g, 'h>>; 3]> {
        assert!(x.is_in_graph(self));
        assert!(scale.is_in_graph(self));
        if let Some(bias) = &bias {
            assert!(bias.is_in_graph(self));
        }
        if let Some(epsilon) = &epsilon {
            assert!(epsilon.is_in_graph(self));
        }
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = [ptr::null_mut(); 3];
        check(unsafe {
            sys::cudnn_fe_graph_rmsnorm(
                self.raw,
                x.as_ptr(),
                scale.as_ptr(),
                opt_ptr(bias),
                opt_ptr(epsilon),
                &raw,
                out.as_mut_ptr(),
                out.len(),
            )
        })?;
        Ok(unsafe { collect_outputs(out, self) })
    }

    /// RMSNorm backward: returns `[dx, dscale, dbias]` (`dbias` per `has_dbias`).
    pub fn rmsnorm_backward<'g>(
        &'g self,
        dy: Tensor<'g, 'h>,
        x: Tensor<'g, 'h>,
        scale: Tensor<'g, 'h>,
        inv_variance: Option<Tensor<'g, 'h>>,
        attrs: &NormBackwardAttributes,
    ) -> Result<[Option<Tensor<'g, 'h>>; 3]> {
        assert!(dy.is_in_graph(self));
        assert!(x.is_in_graph(self));
        assert!(scale.is_in_graph(self));
        if let Some(inv_variance) = &inv_variance {
            assert!(inv_variance.is_in_graph(self));
        }
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = [ptr::null_mut(); 3];
        check(unsafe {
            sys::cudnn_fe_graph_rmsnorm_backward(
                self.raw,
                dy.as_ptr(),
                x.as_ptr(),
                scale.as_ptr(),
                opt_ptr(inv_variance),
                &raw,
                out.as_mut_ptr(),
                out.len(),
            )
        })?;
        Ok(unsafe { collect_outputs(out, self) })
    }

    /// Batchnorm forward (training): `[y, mean, inv_variance, next_running_mean, next_running_var]`.
    #[allow(clippy::too_many_arguments)]
    pub fn batchnorm<'g>(
        &'g self,
        x: Tensor<'g, 'h>,
        scale: Tensor<'g, 'h>,
        bias: Tensor<'g, 'h>,
        in_running_mean: Tensor<'g, 'h>,
        in_running_var: Tensor<'g, 'h>,
        epsilon: Tensor<'g, 'h>,
        momentum: Tensor<'g, 'h>,
        peer_stats: &[Tensor<'g, 'h>],
        attrs: &NormAttributes,
    ) -> Result<[Option<Tensor<'g, 'h>>; 5]> {
        assert!(x.is_in_graph(self));
        assert!(scale.is_in_graph(self));
        assert!(bias.is_in_graph(self));
        assert!(in_running_mean.is_in_graph(self));
        assert!(in_running_var.is_in_graph(self));
        assert!(epsilon.is_in_graph(self));
        assert!(momentum.is_in_graph(self));
        assert!(peer_stats.iter().all(|t| t.is_in_graph(self)));

        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let peers: Vec<*const sys::cudnnFeTensor> = peer_stats.iter().map(|t| t.as_ptr()).collect();
        let mut out = [ptr::null_mut(); 5];
        check(unsafe {
            sys::cudnn_fe_graph_batchnorm(
                self.raw,
                x.as_ptr(),
                scale.as_ptr(),
                bias.as_ptr(),
                in_running_mean.as_ptr(),
                in_running_var.as_ptr(),
                epsilon.as_ptr(),
                momentum.as_ptr(),
                peers.as_ptr(),
                peers.len(),
                &raw,
                out.as_mut_ptr(),
                out.len(),
            )
        })?;
        Ok(unsafe { collect_outputs(out, self) })
    }

    /// Batchnorm inference: returns `y`.
    pub fn batchnorm_inference<'g>(
        &'g self,
        x: Tensor<'g, 'h>,
        mean: Tensor<'g, 'h>,
        inv_variance: Tensor<'g, 'h>,
        scale: Tensor<'g, 'h>,
        bias: Tensor<'g, 'h>,
        attrs: &NormAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(x.is_in_graph(self));
        assert!(mean.is_in_graph(self));
        assert!(inv_variance.is_in_graph(self));
        assert!(scale.is_in_graph(self));
        assert!(bias.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_batchnorm_inference(
                self.raw,
                x.as_ptr(),
                mean.as_ptr(),
                inv_variance.as_ptr(),
                scale.as_ptr(),
                bias.as_ptr(),
                &raw,
                &mut out,
            )
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }

    /// Batchnorm backward: returns `[dx, dscale, dbias]`.
    #[allow(clippy::too_many_arguments)]
    pub fn batchnorm_backward<'g>(
        &'g self,
        dy: Tensor<'g, 'h>,
        x: Tensor<'g, 'h>,
        scale: Tensor<'g, 'h>,
        mean: Tensor<'g, 'h>,
        inv_variance: Tensor<'g, 'h>,
        peer_stats: &[Tensor<'g, 'h>],
        attrs: &NormBackwardAttributes,
    ) -> Result<[Option<Tensor<'g, 'h>>; 3]> {
        assert!(dy.is_in_graph(self));
        assert!(x.is_in_graph(self));
        assert!(scale.is_in_graph(self));
        assert!(mean.is_in_graph(self));
        assert!(inv_variance.is_in_graph(self));
        assert!(peer_stats.iter().all(|t| t.is_in_graph(self)));

        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let peers: Vec<*const sys::cudnnFeTensor> = peer_stats.iter().map(|t| t.as_ptr()).collect();
        let mut out = [ptr::null_mut(); 3];
        check(unsafe {
            sys::cudnn_fe_graph_batchnorm_backward(
                self.raw,
                dy.as_ptr(),
                x.as_ptr(),
                scale.as_ptr(),
                mean.as_ptr(),
                inv_variance.as_ptr(),
                peers.as_ptr(),
                peers.len(),
                &raw,
                out.as_mut_ptr(),
                out.len(),
            )
        })?;
        Ok(unsafe { collect_outputs(out, self) })
    }

    /// Genstats: returns `[sum, sq_sum]`.
    pub fn genstats<'g>(
        &'g self,
        x: Tensor,
        attrs: &NormAttributes,
    ) -> Result<[Option<Tensor<'g, 'h>>; 2]> {
        assert!(x.is_in_graph(self));

        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = [ptr::null_mut(); 2];
        check(unsafe {
            sys::cudnn_fe_graph_genstats(self.raw, x.as_ptr(), &raw, out.as_mut_ptr(), out.len())
        })?;
        Ok(unsafe { collect_outputs(out, self) })
    }

    /// dbn_weight: returns `[dscale, dbias, eq_bias, eq_scale_dy, eq_scale_x]`.
    pub fn dbn_weight<'g>(
        &'g self,
        dy: Tensor,
        x: Tensor,
        scale: Tensor,
        mean: Tensor,
        inv_variance: Tensor,
        attrs: &NormAttributes,
    ) -> Result<[Option<Tensor<'g, 'h>>; 5]> {
        assert!(dy.is_in_graph(self));
        assert!(x.is_in_graph(self));
        assert!(scale.is_in_graph(self));
        assert!(mean.is_in_graph(self));
        assert!(inv_variance.is_in_graph(self));

        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = [ptr::null_mut(); 5];
        check(unsafe {
            sys::cudnn_fe_graph_dbn_weight(
                self.raw,
                dy.as_ptr(),
                x.as_ptr(),
                scale.as_ptr(),
                mean.as_ptr(),
                inv_variance.as_ptr(),
                &raw,
                out.as_mut_ptr(),
                out.len(),
            )
        })?;
        Ok(unsafe { collect_outputs(out, self) })
    }

    /// bn_finalize: returns `[eq_scale, eq_bias, mean, inv_variance, next_running_mean, next_running_var]`.
    #[allow(clippy::too_many_arguments)]
    pub fn bn_finalize<'g>(
        &'g self,
        sum: Tensor,
        sq_sum: Tensor,
        scale: Tensor,
        bias: Tensor,
        epsilon: Tensor,
        accum_count: Tensor,
        attrs: &NormAttributes,
    ) -> Result<[Option<Tensor<'g, 'h>>; 6]> {
        assert!(sum.is_in_graph(self));
        assert!(sq_sum.is_in_graph(self));
        assert!(scale.is_in_graph(self));
        assert!(bias.is_in_graph(self));
        assert!(epsilon.is_in_graph(self));
        assert!(accum_count.is_in_graph(self));

        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = [ptr::null_mut(); 6];
        check(unsafe {
            sys::cudnn_fe_graph_bn_finalize(
                self.raw,
                sum.as_ptr(),
                sq_sum.as_ptr(),
                scale.as_ptr(),
                bias.as_ptr(),
                epsilon.as_ptr(),
                accum_count.as_ptr(),
                &raw,
                out.as_mut_ptr(),
                out.len(),
            )
        })?;
        Ok(unsafe { collect_outputs(out, self) })
    }
}

#[derive(Clone, Default)]
pub struct RoPEAttributes {
    pub compute_data_type: Option<DataType>,
    pub output_scale: Option<f32>,
    pub rope_dim: Option<i64>,
    pub name: Option<String>,
}
impl RoPEAttributes {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn output_scale(mut self, s: f32) -> Self {
        self.output_scale = Some(s);
        self
    }
    pub fn rope_dim(mut self, d: i64) -> Self {
        self.rope_dim = Some(d);
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
    fn to_raw(&self, name: *const std::os::raw::c_char) -> sys::cudnnFeRoPEAttrs {
        sys::cudnnFeRoPEAttrs {
            compute_data_type: self.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            has_output_scale: self.output_scale.is_some() as i32,
            output_scale: self.output_scale.unwrap_or(1.0),
            has_rope_dim: self.rope_dim.is_some() as i32,
            rope_dim: self.rope_dim.unwrap_or(0),
            name,
        }
    }
}

impl<'h> GraphBuilder<'h> {
    pub fn rope<'g>(
        &'g self,
        x: Tensor,
        freqs: Tensor,
        attrs: &RoPEAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(x.is_in_graph(self));
        assert!(freqs.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_rope(self.raw, x.as_ptr(), freqs.as_ptr(), &raw, &mut out)
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
    pub fn rope_backward<'g>(
        &'g self,
        dy: Tensor,
        freqs: Tensor,
        attrs: &RoPEAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(dy.is_in_graph(self));
        assert!(freqs.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = attrs.to_raw(name);
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_rope_backward(self.raw, dy.as_ptr(), freqs.as_ptr(), &raw, &mut out)
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
}

#[derive(Clone, Default)]
pub struct SdpaAttributes<'g, 'h> {
    pub compute_data_type: Option<DataType>,
    pub generate_stats: bool,
    pub attn_scale: Option<f32>,
    pub use_alibi_mask: bool,
    pub use_padding_mask: bool,
    pub use_causal_mask: bool,
    pub use_causal_mask_bottom_right: bool,
    pub diagonal_alignment: Option<DiagonalAlignment>,
    pub diagonal_band_left: Option<i32>,
    pub diagonal_band_right: Option<i32>,
    pub sliding_window_length: Option<i32>,
    pub dropout_probability: Option<f32>,
    pub implementation: Option<AttentionImplementation>,
    pub name: Option<String>,
    pub bias: Option<Tensor<'g, 'h>>,
    pub seq_len_q: Option<Tensor<'g, 'h>>,
    pub seq_len_kv: Option<Tensor<'g, 'h>>,
    pub dropout_seed: Option<Tensor<'g, 'h>>,
    pub dropout_offset: Option<Tensor<'g, 'h>>,
}

impl<'g, 'h> SdpaAttributes<'g, 'h> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn generate_stats(mut self, v: bool) -> Self {
        self.generate_stats = v;
        self
    }
    pub fn attn_scale(mut self, s: f32) -> Self {
        self.attn_scale = Some(s);
        self
    }
    pub fn causal_mask(mut self, v: bool) -> Self {
        self.use_causal_mask = v;
        self
    }
    pub fn causal_mask_bottom_right(mut self, v: bool) -> Self {
        self.use_causal_mask_bottom_right = v;
        self
    }
    pub fn padding_mask(mut self, v: bool) -> Self {
        self.use_padding_mask = v;
        self
    }
    pub fn alibi_mask(mut self, v: bool) -> Self {
        self.use_alibi_mask = v;
        self
    }
    pub fn sliding_window_length(mut self, v: i32) -> Self {
        self.sliding_window_length = Some(v);
        self
    }
    pub fn bias(mut self, t: Tensor<'g, 'h>) -> Self {
        self.bias = Some(t);
        self
    }
    pub fn seq_len(mut self, q: Tensor<'g, 'h>, kv: Tensor<'g, 'h>) -> Self {
        self.seq_len_q = Some(q);
        self.seq_len_kv = Some(kv);
        self
    }
    pub fn dropout(
        mut self,
        probability: f32,
        seed: Tensor<'g, 'h>,
        offset: Tensor<'g, 'h>,
    ) -> Self {
        self.dropout_probability = Some(probability);
        self.dropout_seed = Some(seed);
        self.dropout_offset = Some(offset);
        self
    }
    pub fn implementation(mut self, i: AttentionImplementation) -> Self {
        self.implementation = Some(i);
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

impl<'h> GraphBuilder<'h> {
    /// Scaled dot-product attention forward. Returns `[o, stats]` (`stats` is
    /// `Some` only when `generate_stats`).
    pub fn sdpa<'g>(
        &'g self,
        q: Tensor<'g, 'h>,
        k: Tensor<'g, 'h>,
        v: Tensor<'g, 'h>,
        attrs: &SdpaAttributes<'g, 'h>,
    ) -> Result<[Option<Tensor<'g, 'h>>; 2]> {
        assert!(q.is_in_graph(self));
        assert!(k.is_in_graph(self));
        assert!(v.is_in_graph(self));
        if let Some(bias) = &attrs.bias {
            assert!(bias.is_in_graph(self));
        }
        if let Some(seq_len_q) = &attrs.seq_len_q {
            assert!(seq_len_q.is_in_graph(self));
        }
        if let Some(seq_len_kv) = &attrs.seq_len_kv {
            assert!(seq_len_kv.is_in_graph(self));
        }
        if let Some(dropout_seed) = &attrs.dropout_seed {
            assert!(dropout_seed.is_in_graph(self));
        }
        if let Some(dropout_offset) = &attrs.dropout_offset {
            assert!(dropout_offset.is_in_graph(self));
        }
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeSdpaAttrs {
            compute_data_type: attrs.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            generate_stats: attrs.generate_stats as i32,
            has_attn_scale: attrs.attn_scale.is_some() as i32,
            attn_scale: attrs.attn_scale.unwrap_or(0.0),
            use_alibi_mask: attrs.use_alibi_mask as i32,
            use_padding_mask: attrs.use_padding_mask as i32,
            use_causal_mask: attrs.use_causal_mask as i32,
            use_causal_mask_bottom_right: attrs.use_causal_mask_bottom_right as i32,
            has_diagonal_alignment: attrs.diagonal_alignment.is_some() as i32,
            diagonal_alignment: attrs
                .diagonal_alignment
                .unwrap_or(DiagonalAlignment::TopLeft)
                .to_ffi(),
            has_diagonal_band_left: attrs.diagonal_band_left.is_some() as i32,
            diagonal_band_left: attrs.diagonal_band_left.unwrap_or(0),
            has_diagonal_band_right: attrs.diagonal_band_right.is_some() as i32,
            diagonal_band_right: attrs.diagonal_band_right.unwrap_or(0),
            has_sliding_window: attrs.sliding_window_length.is_some() as i32,
            sliding_window_length: attrs.sliding_window_length.unwrap_or(0),
            has_dropout_prob: attrs.dropout_probability.is_some() as i32,
            dropout_probability: attrs.dropout_probability.unwrap_or(0.0),
            implementation: attrs
                .implementation
                .unwrap_or(AttentionImplementation::Auto)
                .to_ffi(),
            name,
            bias: opt_ptr(attrs.bias),
            seq_len_q: opt_ptr(attrs.seq_len_q),
            seq_len_kv: opt_ptr(attrs.seq_len_kv),
            dropout_seed: opt_ptr(attrs.dropout_seed),
            dropout_offset: opt_ptr(attrs.dropout_offset),
        };
        let mut out = [ptr::null_mut(); 2];
        check(unsafe {
            sys::cudnn_fe_graph_sdpa(
                self.raw,
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                &raw,
                out.as_mut_ptr(),
                out.len(),
            )
        })?;
        Ok(unsafe { collect_outputs(out, self) })
    }
}

/// Attributes for [`Graph::sdpa_backward`].
#[derive(Clone, Default)]
pub struct SdpaBackwardAttributes<'g, 'h> {
    pub compute_data_type: Option<DataType>,
    pub attn_scale: Option<f32>,
    pub use_alibi_mask: bool,
    pub use_padding_mask: bool,
    pub use_causal_mask: bool,
    pub use_causal_mask_bottom_right: bool,
    pub diagonal_alignment: Option<DiagonalAlignment>,
    pub diagonal_band_left: Option<i32>,
    pub diagonal_band_right: Option<i32>,
    pub sliding_window_length: Option<i32>,
    pub dropout_probability: Option<f32>,
    pub use_deterministic_algorithm: bool,
    pub name: Option<String>,
    pub bias: Option<Tensor<'g, 'h>>,
    pub dbias: Option<Tensor<'g, 'h>>,
    pub seq_len_q: Option<Tensor<'g, 'h>>,
    pub seq_len_kv: Option<Tensor<'g, 'h>>,
    pub dropout_seed: Option<Tensor<'g, 'h>>,
    pub dropout_offset: Option<Tensor<'g, 'h>>,
}

impl<'g, 'h> SdpaBackwardAttributes<'g, 'h> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn attn_scale(mut self, s: f32) -> Self {
        self.attn_scale = Some(s);
        self
    }
    pub fn causal_mask(mut self, v: bool) -> Self {
        self.use_causal_mask = v;
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

impl<'h> GraphBuilder<'h> {
    /// Scaled dot-product attention backward. Returns `[dq, dk, dv]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa_backward<'g>(
        &'g self,
        q: Tensor<'g, 'h>,
        k: Tensor<'g, 'h>,
        v: Tensor<'g, 'h>,
        o: Tensor<'g, 'h>,
        d_o: Tensor<'g, 'h>,
        stats: Tensor<'g, 'h>,
        attrs: &SdpaBackwardAttributes<'g, 'h>,
    ) -> Result<[Option<Tensor<'g, 'h>>; 3]> {
        assert!(q.is_in_graph(self));
        assert!(k.is_in_graph(self));
        assert!(v.is_in_graph(self));
        assert!(o.is_in_graph(self));
        assert!(d_o.is_in_graph(self));
        assert!(stats.is_in_graph(self));
        if let Some(bias) = &attrs.bias {
            assert!(bias.is_in_graph(self));
        }
        if let Some(dbias) = &attrs.dbias {
            assert!(dbias.is_in_graph(self));
        }
        if let Some(seq_len_q) = &attrs.seq_len_q {
            assert!(seq_len_q.is_in_graph(self));
        }
        if let Some(seq_len_kv) = &attrs.seq_len_kv {
            assert!(seq_len_kv.is_in_graph(self));
        }
        if let Some(dropout_seed) = &attrs.dropout_seed {
            assert!(dropout_seed.is_in_graph(self));
        }
        if let Some(dropout_offset) = &attrs.dropout_offset {
            assert!(dropout_offset.is_in_graph(self));
        }
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeSdpaBackwardAttrs {
            compute_data_type: attrs.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            has_attn_scale: attrs.attn_scale.is_some() as i32,
            attn_scale: attrs.attn_scale.unwrap_or(0.0),
            use_alibi_mask: attrs.use_alibi_mask as i32,
            use_padding_mask: attrs.use_padding_mask as i32,
            use_causal_mask: attrs.use_causal_mask as i32,
            use_causal_mask_bottom_right: attrs.use_causal_mask_bottom_right as i32,
            has_diagonal_alignment: attrs.diagonal_alignment.is_some() as i32,
            diagonal_alignment: attrs
                .diagonal_alignment
                .unwrap_or(DiagonalAlignment::TopLeft)
                .to_ffi(),
            has_diagonal_band_left: attrs.diagonal_band_left.is_some() as i32,
            diagonal_band_left: attrs.diagonal_band_left.unwrap_or(0),
            has_diagonal_band_right: attrs.diagonal_band_right.is_some() as i32,
            diagonal_band_right: attrs.diagonal_band_right.unwrap_or(0),
            has_sliding_window: attrs.sliding_window_length.is_some() as i32,
            sliding_window_length: attrs.sliding_window_length.unwrap_or(0),
            has_dropout_prob: attrs.dropout_probability.is_some() as i32,
            dropout_probability: attrs.dropout_probability.unwrap_or(0.0),
            use_deterministic_algorithm: attrs.use_deterministic_algorithm as i32,
            name,
            bias: opt_ptr(attrs.bias),
            dbias: opt_ptr(attrs.dbias),
            seq_len_q: opt_ptr(attrs.seq_len_q),
            seq_len_kv: opt_ptr(attrs.seq_len_kv),
            dropout_seed: opt_ptr(attrs.dropout_seed),
            dropout_offset: opt_ptr(attrs.dropout_offset),
        };
        let mut out = [ptr::null_mut(); 3];
        check(unsafe {
            sys::cudnn_fe_graph_sdpa_backward(
                self.raw,
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                o.as_ptr(),
                d_o.as_ptr(),
                stats.as_ptr(),
                &raw,
                out.as_mut_ptr(),
                out.len(),
            )
        })?;
        Ok(unsafe { collect_outputs(out, self) })
    }
}

/* ================================================================ */
/* Quantization                                                     */
/* ================================================================ */

#[derive(Clone, Default)]
pub struct BlockScaleQuantizeAttributes {
    pub compute_data_type: Option<DataType>,
    pub block_size: i32,
    pub axis: Option<i64>,
    pub transpose: Option<bool>,
    pub name: Option<String>,
}
impl BlockScaleQuantizeAttributes {
    pub fn new(block_size: i32) -> Self {
        BlockScaleQuantizeAttributes {
            block_size,
            ..Default::default()
        }
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn axis(mut self, a: i64) -> Self {
        self.axis = Some(a);
        self
    }
    pub fn transpose(mut self, v: bool) -> Self {
        self.transpose = Some(v);
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

#[derive(Clone, Default)]
pub struct BlockScaleDequantizeAttributes {
    pub compute_data_type: Option<DataType>,
    pub block_size: Vec<i32>,
    pub is_negative_scale: Option<bool>,
    pub name: Option<String>,
}
impl BlockScaleDequantizeAttributes {
    pub fn new(block_size: &[i32]) -> Self {
        BlockScaleDequantizeAttributes {
            block_size: block_size.to_vec(),
            ..Default::default()
        }
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn is_negative_scale(mut self, v: bool) -> Self {
        self.is_negative_scale = Some(v);
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

impl<'h> GraphBuilder<'h> {
    /// Block-scale quantize: returns `[y, scale]`.
    pub fn block_scale_quantize<'g>(
        &'g self,
        x: Tensor<'g, 'h>,
        attrs: &BlockScaleQuantizeAttributes,
    ) -> Result<[Option<Tensor<'g, 'h>>; 2]> {
        assert!(x.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeBlockScaleQuantizeAttrs {
            compute_data_type: attrs.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            block_size: attrs.block_size,
            has_axis: attrs.axis.is_some() as i32,
            axis: attrs.axis.unwrap_or(0),
            has_transpose: attrs.transpose.is_some() as i32,
            transpose: attrs.transpose.unwrap_or(false) as i32,
            name,
        };
        let mut out = [ptr::null_mut(); 2];
        check(unsafe {
            sys::cudnn_fe_graph_block_scale_quantize(
                self.raw,
                x.as_ptr(),
                &raw,
                out.as_mut_ptr(),
                out.len(),
            )
        })?;
        Ok(unsafe { collect_outputs(out, self) })
    }

    pub fn block_scale_dequantize<'g>(
        &'g self,
        x: Tensor<'g, 'h>,
        scale: Tensor<'g, 'h>,
        attrs: &BlockScaleDequantizeAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(x.is_in_graph(self));
        assert!(scale.is_in_graph(self));
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeBlockScaleDequantizeAttrs {
            compute_data_type: attrs.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            block_size: attrs.block_size.as_ptr(),
            block_size_len: attrs.block_size.len(),
            has_is_negative_scale: attrs.is_negative_scale.is_some() as i32,
            is_negative_scale: attrs.is_negative_scale.unwrap_or(false) as i32,
            name,
        };
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_block_scale_dequantize(
                self.raw,
                x.as_ptr(),
                scale.as_ptr(),
                &raw,
                &mut out,
            )
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
}

/* ================================================================ */
/* MoE grouped matmul                                               */
/* ================================================================ */

#[derive(Clone, Default)]
pub struct MoeGroupedMatmulAttributes {
    pub compute_data_type: Option<DataType>,
    pub mode: Option<MoeGroupedMatmulMode>,
    pub top_k: Option<i32>,
    pub name: Option<String>,
}
impl MoeGroupedMatmulAttributes {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn compute_data_type(mut self, dt: DataType) -> Self {
        self.compute_data_type = Some(dt);
        self
    }
    pub fn mode(mut self, m: MoeGroupedMatmulMode) -> Self {
        self.mode = Some(m);
        self
    }
    pub fn top_k(mut self, k: i32) -> Self {
        self.top_k = Some(k);
        self
    }
    pub fn name(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
}

impl<'h> GraphBuilder<'h> {
    /// `token_index` and `token_ks` are optional.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_grouped_matmul<'g>(
        &'g self,
        token: Tensor<'g, 'h>,
        weight: Tensor<'g, 'h>,
        first_token_offset: Tensor<'g, 'h>,
        token_index: Option<Tensor<'g, 'h>>,
        token_ks: Option<Tensor<'g, 'h>>,
        attrs: &MoeGroupedMatmulAttributes,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(token.is_in_graph(self));
        assert!(weight.is_in_graph(self));
        assert!(first_token_offset.is_in_graph(self));
        if let Some(token_index) = &token_index {
            assert!(token_index.is_in_graph(self));
        }
        if let Some(token_ks) = &token_ks {
            assert!(token_ks.is_in_graph(self));
        }
        let (_n, name) = cstr(attrs.name.as_deref());
        let raw = sys::cudnnFeMoeGroupedMatmulAttrs {
            compute_data_type: attrs.compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            mode: attrs.mode.unwrap_or(MoeGroupedMatmulMode::NotSet).to_ffi(),
            has_top_k: attrs.top_k.is_some() as i32,
            top_k: attrs.top_k.unwrap_or(0),
            name,
        };
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_moe_grouped_matmul(
                self.raw,
                token.as_ptr(),
                weight.as_ptr(),
                first_token_offset.as_ptr(),
                opt_ptr(token_index),
                opt_ptr(token_ks),
                &raw,
                &mut out,
            )
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }

    pub fn moe_grouped_matmul_bwd<'g>(
        &'g self,
        doutput: Tensor<'g, 'h>,
        token: Tensor<'g, 'h>,
        first_token_offset: Tensor<'g, 'h>,
        compute_data_type: Option<DataType>,
        name: Option<&str>,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(doutput.is_in_graph(self));
        assert!(token.is_in_graph(self));
        assert!(first_token_offset.is_in_graph(self));
        let (_n, name_p) = cstr(name);
        let raw = sys::cudnnFeMoeGroupedMatmulBwdAttrs {
            compute_data_type: compute_data_type.unwrap_or(DataType::NotSet).to_ffi(),
            name: name_p,
        };
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_moe_grouped_matmul_bwd(
                self.raw,
                doutput.as_ptr(),
                token.as_ptr(),
                first_token_offset.as_ptr(),
                &raw,
                &mut out,
            )
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }
}
