//! Error types for HDF5 format parsing.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::string::String;

#[cfg(feature = "std")]
use std::string::String;

use thiserror::Error;

/// Errors that can occur when parsing HDF5 binary format structures.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FormatError {
    /// The HDF5 magic signature was not found at any valid offset.
    #[error("HDF5 signature not found at any valid offset")]
    SignatureNotFound,
    /// The superblock version is not supported.
    #[error("unsupported superblock version: {0}")]
    UnsupportedVersion(u8),
    /// Unexpected end of data.
    #[error("unexpected EOF: need {expected} bytes, have {available}")]
    UnexpectedEof {
        /// Number of bytes expected.
        expected: usize,
        /// Number of bytes actually available.
        available: usize,
    },
    /// Invalid offset size (must be 2, 4, or 8).
    #[error("invalid offset size: {0} (must be 2, 4, or 8)")]
    InvalidOffsetSize(u8),
    /// Invalid length size (must be 2, 4, or 8).
    #[error("invalid length size: {0} (must be 2, 4, or 8)")]
    InvalidLengthSize(u8),
    /// Invalid object header signature.
    #[error("invalid object header signature")]
    InvalidObjectHeaderSignature,
    /// Invalid object header version.
    #[error("invalid object header version: {0}")]
    InvalidObjectHeaderVersion(u8),
    /// Unknown message type that is marked as must-understand.
    #[error("unsupported message type {0:#06x} marked as must-understand")]
    UnsupportedMessage(u16),
    /// Invalid datatype class.
    #[error("invalid datatype class: {0}")]
    InvalidDatatypeClass(u8),
    /// Invalid datatype version for a given class.
    #[error("invalid datatype version {version} for class {class}")]
    InvalidDatatypeVersion {
        /// The type class.
        class: u8,
        /// The version found.
        version: u8,
    },
    /// Invalid string padding type.
    #[error("invalid string padding type: {0}")]
    InvalidStringPadding(u8),
    /// Invalid character set.
    #[error("invalid character set: {0}")]
    InvalidCharacterSet(u8),
    /// Invalid byte order.
    #[error("invalid byte order: {0}")]
    InvalidByteOrder(u8),
    /// Invalid reference type.
    #[error("invalid reference type: {0}")]
    InvalidReferenceType(u8),
    /// Invalid dataspace version.
    #[error("invalid dataspace version: {0}")]
    InvalidDataspaceVersion(u8),
    /// Invalid dataspace type.
    #[error("invalid dataspace type: {0}")]
    InvalidDataspaceType(u8),
    /// Invalid data layout version.
    #[error("invalid data layout version: {0}")]
    InvalidLayoutVersion(u8),
    /// Invalid data layout class.
    #[error("invalid data layout class: {0}")]
    InvalidLayoutClass(u8),
    /// No data allocated for contiguous layout.
    #[error("no data allocated for contiguous layout")]
    NoDataAllocated,
    /// Type mismatch when reading data.
    #[error("type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        /// Expected type description.
        expected: &'static str,
        /// Actual type description.
        actual: &'static str,
    },
    /// Data size mismatch.
    #[error("data size mismatch: expected {expected} bytes, got {actual} bytes")]
    DataSizeMismatch {
        /// Expected size in bytes.
        expected: usize,
        /// Actual size in bytes.
        actual: usize,
    },
    /// Invalid local heap signature.
    #[error("invalid local heap signature")]
    InvalidLocalHeapSignature,
    /// Invalid local heap version.
    #[error("invalid local heap version: {0}")]
    InvalidLocalHeapVersion(u8),
    /// Invalid B-tree v1 signature.
    #[error("invalid B-tree v1 signature")]
    InvalidBTreeSignature,
    /// Invalid B-tree node type.
    #[error("invalid B-tree node type: {0}")]
    InvalidBTreeNodeType(u8),
    /// Invalid symbol table node signature.
    #[error("invalid symbol table node signature")]
    InvalidSymbolTableNodeSignature,
    /// Invalid symbol table node version.
    #[error("invalid symbol table node version: {0}")]
    InvalidSymbolTableNodeVersion(u8),
    /// Path not found during group traversal.
    #[error("path not found: {0}")]
    PathNotFound(String),
    /// Invalid Link message version.
    #[error("invalid link message version: {0}")]
    InvalidLinkVersion(u8),
    /// Invalid link type code.
    #[error("invalid link type: {0}")]
    InvalidLinkType(u8),
    /// Invalid Link Info message version.
    #[error("invalid link info message version: {0}")]
    InvalidLinkInfoVersion(u8),
    /// Invalid Group Info message version.
    #[error("invalid group info message version: {0}")]
    InvalidGroupInfoVersion(u8),
    /// Invalid B-tree v2 signature.
    #[error("invalid B-tree v2 signature")]
    InvalidBTreeV2Signature,
    /// Invalid B-tree v2 version.
    #[error("invalid B-tree v2 version: {0}")]
    InvalidBTreeV2Version(u8),
    /// Invalid fractal heap signature.
    #[error("invalid fractal heap signature")]
    InvalidFractalHeapSignature,
    /// Invalid fractal heap version.
    #[error("invalid fractal heap version: {0}")]
    InvalidFractalHeapVersion(u8),
    /// Invalid heap ID type.
    #[error("invalid heap ID type: {0}")]
    InvalidHeapIdType(u8),
    /// Invalid attribute message version.
    #[error("invalid attribute message version: {0}")]
    InvalidAttributeVersion(u8),
    /// Invalid Attribute Info message version.
    #[error("invalid attribute info message version: {0}")]
    InvalidAttributeInfoVersion(u8),
    /// Invalid shared message version.
    #[error("invalid shared message version: {0}")]
    InvalidSharedMessageVersion(u8),
    /// Invalid SOHM table version.
    #[error("invalid SOHM table version: {0}")]
    InvalidSohmTableVersion(u8),
    /// Invalid SOHM table signature (expected "SMTB").
    #[error("invalid SOHM table signature (expected SMTB)")]
    InvalidSohmTableSignature,
    /// Invalid SOHM list signature (expected "SMLI").
    #[error("invalid SOHM list signature (expected SMLI)")]
    InvalidSohmListSignature,
    /// Invalid global heap collection signature.
    #[error("invalid global heap collection signature")]
    InvalidGlobalHeapSignature,
    /// Invalid global heap version.
    #[error("invalid global heap version: {0}")]
    InvalidGlobalHeapVersion(u8),
    /// Global heap object not found.
    #[error("global heap object not found: collection {collection_address:#x}, index {index}")]
    GlobalHeapObjectNotFound {
        /// Address of the collection.
        collection_address: u64,
        /// Index that was not found.
        index: u16,
    },
    /// Variable-length data error.
    #[error("variable-length data error: {0}")]
    VlDataError(String),
    /// Serialization error.
    #[error("serialization error: {0}")]
    SerializationError(String),
    /// Dataset is missing data.
    #[error("dataset is missing data")]
    DatasetMissingData,
    /// Dataset is missing shape.
    #[error("dataset is missing shape")]
    DatasetMissingShape,
    /// Invalid filter pipeline version.
    #[error("invalid filter pipeline version: {0}")]
    InvalidFilterPipelineVersion(u8),
    /// Unsupported filter ID.
    #[error("unsupported filter: {0}")]
    UnsupportedFilter(u16),
    /// Filter processing error.
    #[error("filter error: {0}")]
    FilterError(String),
    /// Decompression error.
    #[error("decompression error: {0}")]
    DecompressionError(String),
    /// Compression error.
    #[error("compression error: {0}")]
    CompressionError(String),
    /// Fletcher32 checksum mismatch.
    #[error("fletcher32 mismatch: expected {expected:#010x}, computed {computed:#010x}")]
    Fletcher32Mismatch {
        /// Expected checksum.
        expected: u32,
        /// Computed checksum.
        computed: u32,
    },
    /// Chunked dataset read error.
    #[error("chunked read error: {0}")]
    ChunkedReadError(String),
    /// Chunk assembly error.
    #[error("chunk assembly error: {0}")]
    ChunkAssemblyError(String),
    /// CRC32C checksum mismatch.
    #[error("checksum mismatch: expected {expected:#010x}, computed {computed:#010x}")]
    ChecksumMismatch {
        /// The checksum stored in the file.
        expected: u32,
        /// The checksum we computed.
        computed: u32,
    },
    /// Maximum nesting/continuation depth exceeded (malformed data protection).
    #[error("maximum nesting/continuation depth exceeded")]
    NestingDepthExceeded,
    /// Duplicate dataset name detected during parallel metadata merge.
    #[error("duplicate dataset name during parallel merge: {0}")]
    DuplicateDatasetName(String),
    /// General error message
    #[error("Error: {0}")]
    GeneralError(String),
}

impl FormatError {
    pub fn stop(e: &'static str) -> Self {
        Self::GeneralError(e.to_string())
    }
}