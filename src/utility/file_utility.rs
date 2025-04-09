/// Converts floats to bytes, using Big Endian format
pub fn floats_to_bytes(vec: Vec<f64>) -> Vec<u8> {
    vec.iter().map(|x| x.to_be_bytes()).flatten().collect()
}

/// Converts a vector of bytes to floats, using Big Endian
pub fn bytes_to_floats(vec: Vec<u8>) -> Vec<f64> {
    let mut floats = vec![0f64; vec.len() / 8];

    for i in 0..floats.len() {
        floats[i] = f64::from_be_bytes(vec[(i * 8)..((i + 1) * 8)].try_into().unwrap())
    }

    floats
}

/// Converts u64s to bytes, using Big Endian format
pub fn u64s_to_bytes(vec: Vec<u64>) -> Vec<u8> {
    vec.iter().map(|x| x.to_be_bytes()).flatten().collect()
}

/// Converts a vector of bytes to u64s, using Big Endian
pub fn bytes_to_u64s(vec: Vec<u8>) -> Vec<u64> {
    let mut ints = vec![0u64; vec.len() / 8];

    for i in 0..ints.len() {
        ints[i] = u64::from_be_bytes(vec[(i * 8)..((i + 1) * 8)].try_into().unwrap())
    }

    ints
}


#[cfg(test)]
mod file_tests {
    use crate::utility::file_utility::{
        bytes_to_floats, bytes_to_u64s, floats_to_bytes, u64s_to_bytes,
    };

    #[test]
    fn test_byte_conversion() {
        let float_vec = vec![0.3453, 0.3467245372, 123513.1462456257752];
        debug_assert_eq!(
            float_vec,
            bytes_to_floats(floats_to_bytes(float_vec.clone()))
        );

        let int_vec = vec![23532, 6246, 0000, 0465345];
        debug_assert_eq!(int_vec, bytes_to_u64s(u64s_to_bytes(int_vec.clone())));
    }
}
