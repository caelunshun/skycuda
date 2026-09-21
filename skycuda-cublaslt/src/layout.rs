// --- LLM-generated --- //

//! Matrix shape/layout description and the owned cuBLASLt descriptor built from it.

use crate::{
    enums::{DataType, Order},
    error::{Result, check},
    sys,
};
use std::ptr;

/// Shape, element type and memory order of one matmul operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatrixLayout {
    pub data_type: DataType,
    pub rows: u64,
    pub cols: u64,
    /// Leading dimension, in elements.
    pub ld: i64,
    pub order: Order,
    pub batch_count: i32,
    /// Stride between batch entries, in elements.
    pub batch_stride: i64,
}

impl MatrixLayout {
    /// A single column-major matrix.
    pub fn new(data_type: DataType, rows: u64, cols: u64, ld: i64) -> Self {
        Self {
            data_type,
            rows,
            cols,
            ld,
            order: Order::Col,
            batch_count: 1,
            batch_stride: 0,
        }
    }

    pub fn order(mut self, order: Order) -> Self {
        self.order = order;
        self
    }

    pub fn batched(mut self, count: i32, stride: i64) -> Self {
        self.batch_count = count;
        self.batch_stride = stride;
        self
    }

    /// Smallest allocation, in bytes, that holds every element this layout
    /// addresses.
    ///
    /// `None` when the extent cannot be reasoned about: the `COL32*` orders,
    /// which pack elements into hardware-specific tiles, and negative
    /// `ld`/`batch_stride`, which some algorithms accept but whose base pointer
    /// no longer sits at the start of the allocation.
    pub fn min_len_bytes(&self) -> Option<usize> {
        if !matches!(self.order, Order::Col | Order::Row)
            || self.ld < 0
            || self.batch_stride < 0
            || self.batch_count < 1
        {
            return None;
        }
        if self.rows == 0 || self.cols == 0 {
            return Some(0);
        }

        // Extent of the last element of the last batch entry, in elements.
        let (major, minor) = match self.order {
            Order::Row => (self.rows, self.cols),
            _ => (self.cols, self.rows),
        };
        let ld = self.ld as u64;
        let batch = (self.batch_count as u64 - 1).checked_mul(self.batch_stride as u64)?;
        let elements = (major - 1)
            .checked_mul(ld)?
            .checked_add(minor)?
            .checked_add(batch)?;

        usize::try_from(
            elements
                .checked_mul(self.data_type.size_bits() as u64)?
                .div_ceil(8),
        )
        .ok()
    }
}

/// An owned `cublasLtMatrixLayout_t`.
#[derive(Debug)]
pub(crate) struct RawMatrixLayout(sys::cublasLtMatrixLayout_t);

impl RawMatrixLayout {
    pub fn new(layout: &MatrixLayout) -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(
            unsafe {
                sys::cublasLtMatrixLayoutCreate(
                    &mut raw,
                    layout.data_type.to_ffi(),
                    layout.rows,
                    layout.cols,
                    layout.ld,
                )
            },
            "cublasLtMatrixLayoutCreate",
        )?;
        let this = Self(raw);

        use sys::cublasLtMatrixLayoutAttribute_t::*;
        this.set(CUBLASLT_MATRIX_LAYOUT_ORDER, &(layout.order.raw() as i32))?;
        this.set(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &layout.batch_count)?;
        this.set(
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &layout.batch_stride,
        )?;
        Ok(this)
    }

    pub fn raw(&self) -> sys::cublasLtMatrixLayout_t {
        self.0
    }

    fn set<T>(&self, attr: sys::cublasLtMatrixLayoutAttribute_t, value: &T) -> Result<()> {
        check(
            unsafe {
                sys::cublasLtMatrixLayoutSetAttribute(
                    self.0,
                    attr,
                    (value as *const T).cast(),
                    size_of::<T>(),
                )
            },
            "cublasLtMatrixLayoutSetAttribute",
        )
    }
}

impl Drop for RawMatrixLayout {
    fn drop(&mut self) {
        unsafe {
            sys::cublasLtMatrixLayoutDestroy(self.0);
        }
    }
}

// SAFETY: cuBLASLt descriptors are plain value objects; we only mutate them
// behind `&mut` or a mutex.
unsafe impl Send for RawMatrixLayout {}
unsafe impl Sync for RawMatrixLayout {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_len_bytes_col_major() {
        // 4x3 f32, tightly packed.
        let l = MatrixLayout::new(DataType::F32, 4, 3, 4);
        assert_eq!(l.min_len_bytes(), Some(4 * 3 * 4));

        // Padded leading dimension: the last column still only needs `rows`
        // elements, so the trailing pad is not required.
        let l = MatrixLayout::new(DataType::F32, 4, 3, 8);
        assert_eq!(l.min_len_bytes(), Some((2 * 8 + 4) * 4));
    }

    #[test]
    fn min_len_bytes_row_major() {
        let l = MatrixLayout::new(DataType::F16, 4, 3, 3).order(Order::Row);
        assert_eq!(l.min_len_bytes(), Some(4 * 3 * 2));

        let l = MatrixLayout::new(DataType::F16, 4, 3, 6).order(Order::Row);
        assert_eq!(l.min_len_bytes(), Some((3 * 6 + 3) * 2));
    }

    #[test]
    fn min_len_bytes_batched_and_subbyte() {
        let l = MatrixLayout::new(DataType::F32, 4, 3, 4).batched(2, 12);
        assert_eq!(l.min_len_bytes(), Some(2 * 4 * 3 * 4));

        // 4-bit elements round up to whole bytes.
        let l = MatrixLayout::new(DataType::F4E2M1, 3, 3, 3);
        assert_eq!(l.min_len_bytes(), Some((9 * 4_usize).div_ceil(8)));
    }

    #[test]
    fn min_len_bytes_opts_out() {
        assert_eq!(
            MatrixLayout::new(DataType::I8, 4, 4, 32)
                .order(Order::Col32)
                .min_len_bytes(),
            None
        );
        assert_eq!(
            MatrixLayout::new(DataType::F32, 4, 4, -4).min_len_bytes(),
            None
        );
        assert_eq!(
            MatrixLayout::new(DataType::F32, 0, 4, 4).min_len_bytes(),
            Some(0)
        );
    }
}
