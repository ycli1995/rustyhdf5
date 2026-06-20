//! HDF5 object header message type identifiers.

/// Recognized HDF5 header message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Nil,
    Dataspace,
    LinkInfo,
    Datatype,
    FillValueOld,
    FillValue,
    Link,
    DataLayout,
    GroupInfo,
    FilterPipeline,
    Attribute,
    ObjectHeaderContinuation,
    SymbolTable,
    ObjectModificationTime,
    BTreeKValues,
    SharedMessageTable,
    AttributeInfo,
    ObjectReferenceCount,
    /// Unknown message type with its raw type ID.
    Unknown(u16),
}

impl MessageType {
    /// Convert a raw u16 type ID to a `MessageType`.
    pub fn from_u16(val: u16) -> Self {
        match val {
            0x0000 => Self::Nil,
            0x0001 => Self::Dataspace,
            0x0002 => Self::LinkInfo,
            0x0003 => Self::Datatype,
            0x0004 => Self::FillValueOld,
            0x0005 => Self::FillValue,
            0x0006 => Self::Link,
            0x0008 => Self::DataLayout,
            0x000A => Self::GroupInfo,
            0x000B => Self::FilterPipeline,
            0x000C => Self::Attribute,
            0x000F => Self::SharedMessageTable,
            0x0010 => Self::ObjectHeaderContinuation,
            0x0011 => Self::SymbolTable,
            0x0012 => Self::ObjectModificationTime,
            0x0013 => Self::BTreeKValues,
            0x0015 => Self::AttributeInfo,
            0x0016 => Self::ObjectReferenceCount,
            other => Self::Unknown(other),
        }
    }

    /// Convert back to the raw u16 type ID.
    pub fn to_u16(self) -> u16 {
        match self {
            Self::Nil => 0x0000,
            Self::Dataspace => 0x0001,
            Self::LinkInfo => 0x0002,
            Self::Datatype => 0x0003,
            Self::FillValueOld => 0x0004,
            Self::FillValue => 0x0005,
            Self::Link => 0x0006,
            Self::DataLayout => 0x0008,
            Self::GroupInfo => 0x000A,
            Self::FilterPipeline => 0x000B,
            Self::Attribute => 0x000C,
            Self::SharedMessageTable => 0x000F,
            Self::ObjectHeaderContinuation => 0x0010,
            Self::SymbolTable => 0x0011,
            Self::ObjectModificationTime => 0x0012,
            Self::BTreeKValues => 0x0013,
            Self::AttributeInfo => 0x0015,
            Self::ObjectReferenceCount => 0x0016,
            Self::Unknown(v) => v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_types_roundtrip() {
        let known = [
            (0x0000, MessageType::Nil),
            (0x0001, MessageType::Dataspace),
            (0x0002, MessageType::LinkInfo),
            (0x0003, MessageType::Datatype),
            (0x0004, MessageType::FillValueOld),
            (0x0005, MessageType::FillValue),
            (0x0006, MessageType::Link),
            (0x0008, MessageType::DataLayout),
            (0x000A, MessageType::GroupInfo),
            (0x000B, MessageType::FilterPipeline),
            (0x000C, MessageType::Attribute),
            (0x000F, MessageType::SharedMessageTable),
            (0x0010, MessageType::ObjectHeaderContinuation),
            (0x0011, MessageType::SymbolTable),
            (0x0012, MessageType::ObjectModificationTime),
            (0x0013, MessageType::BTreeKValues),
            (0x0015, MessageType::AttributeInfo),
            (0x0016, MessageType::ObjectReferenceCount),
        ];
        for (val, expected) in &known {
            let mt = MessageType::from_u16(*val);
            assert_eq!(mt, *expected);
            assert_eq!(mt.to_u16(), *val);
        }
    }

    #[test]
    fn unknown_type() {
        let mt = MessageType::from_u16(0x00FF);
        assert_eq!(mt, MessageType::Unknown(0x00FF));
        assert_eq!(mt.to_u16(), 0x00FF);
    }

    #[test]
    fn unknown_type_zero_gap() {
        // 0x0007 is not a defined type
        let mt = MessageType::from_u16(0x0007);
        assert_eq!(mt, MessageType::Unknown(0x0007));
    }
}
